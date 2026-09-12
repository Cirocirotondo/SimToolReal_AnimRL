"""ZMQ client for the UR5e low-level controller (``impedance_controller``).

Despite its name that controller is a cartesian/joint velocity servo with PD
behaviour: its joint-mode velocity command is proportional to
``target_q - measured_q``. Two consequences shape this module.

1. Braking is done by repeatedly commanding the *latest measured* position,
   which requests approximately zero velocity. There is no stop message.
2. A target must be streamed continuously, not sent once, or the servo keeps
   chasing a stale setpoint. ``CommandStreamer`` re-sends the current target
   at a fixed rate from a background thread while the policy computes the next
   one at its own, slower, control rate.

Ports come from the controller's JSON config (``pc_ur_new.json``):
``socket_port`` for commands (PUB, bound here) and ``publisher_port`` for state
(SUB, connected here).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import zmq

ARM_DOF = 6
DEFAULT_STREAM_HZ = 100.0
DEFAULT_BRAKE_SECONDS = 0.5
DEFAULT_BRAKE_HZ = 100.0
# ZMQ PUB silently discards anything published before a subscriber has
# finished connecting -- the "slow joiner" problem. Without this pause the
# first message after bind(), which is the homing trajectory, can vanish: the
# operator presses Space, nothing moves, and the policy would then start from
# the wrong pose. The homing tolerance check catches that, but not sending it
# into the void is better.
DEFAULT_CONNECT_SETTLE_SECONDS = 1.0


class ArmClientError(RuntimeError):
    """The arm state stream is absent, stale or malformed."""


class _CommandStreamer:
    def __init__(self, command_socket: zmq.Socket, frequency_hz: float) -> None:
        if frequency_hz <= 0.0:
            raise ValueError("Command stream frequency must be positive")
        self.command_socket = command_socket
        self.period = 1.0 / float(frequency_hz)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.target_q: Optional[list] = None
        self.sent_count = 0
        self.thread = threading.Thread(
            target=self._run, name="animrl-arm-streamer", daemon=True
        )

    def start(self) -> None:
        self.thread.start()

    def set_target(self, target_q: np.ndarray) -> None:
        target = np.asarray(target_q, dtype=np.float64)
        if target.shape != (ARM_DOF,):
            raise ValueError("Arm target must have shape (6,)")
        if not np.all(np.isfinite(target)):
            raise ValueError("Arm target contains non-finite values")
        with self.lock:
            self.target_q = target.tolist()

    def clear_target(self) -> None:
        with self.lock:
            self.target_q = None

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)

    def _run(self) -> None:
        next_send = time.monotonic()
        while not self.stop_event.is_set():
            with self.lock:
                target_q = None if self.target_q is None else list(self.target_q)
            if target_q is not None:
                self.command_socket.send_json({"target_q": target_q})
                self.sent_count += 1
            next_send += self.period
            sleep_dt = next_send - time.monotonic()
            if sleep_dt > 0:
                self.stop_event.wait(sleep_dt)
            else:
                next_send = time.monotonic()


class ArmClient:
    def __init__(
        self,
        controller_config_path: Path,
        *,
        stream_hz: float = DEFAULT_STREAM_HZ,
        state_timeout: float = 0.25,
        connect_settle_seconds: float = DEFAULT_CONNECT_SETTLE_SECONDS,
    ) -> None:
        config_path = Path(controller_config_path).expanduser().resolve()
        if not config_path.is_file():
            raise ArmClientError(
                "UR5 controller config not found: {}".format(config_path)
            )
        with config_path.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
        for key in ("socket_port", "publisher_port"):
            if key not in config:
                raise ArmClientError("{} is missing '{}'".format(config_path, key))
        self.config_path = config_path
        self.state_timeout = float(state_timeout)

        self.context = zmq.Context()
        self.command_socket = self.context.socket(zmq.PUB)
        self.command_socket.bind("tcp://*:{}".format(config["socket_port"]))
        self.state_socket = self.context.socket(zmq.SUB)
        self.state_socket.setsockopt(zmq.CONFLATE, 1)
        self.state_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.state_socket.connect(
            "tcp://127.0.0.1:{}".format(config["publisher_port"])
        )

        self.positions: Optional[np.ndarray] = None
        self.velocities: Optional[np.ndarray] = None
        self.last_state: Optional[dict] = None
        self.last_state_at: Optional[float] = None
        self._streamer = _CommandStreamer(self.command_socket, stream_hz)
        self._streaming = False
        if connect_settle_seconds > 0.0:
            time.sleep(float(connect_settle_seconds))

    # -- receiving -------------------------------------------------------
    def poll(self) -> bool:
        state = None
        while True:
            try:
                state = self.state_socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
        if not isinstance(state, dict):
            return False
        positions = np.asarray(state.get("Q", []), dtype=np.float64)
        if positions.shape[0] < ARM_DOF or not np.all(
            np.isfinite(positions[:ARM_DOF])
        ):
            return False
        velocities = np.asarray(state.get("Qd", []), dtype=np.float64)
        self.positions = positions[:ARM_DOF].copy()
        if velocities.shape[0] >= ARM_DOF and np.all(np.isfinite(velocities[:ARM_DOF])):
            self.velocities = velocities[:ARM_DOF].copy()
        else:
            self.velocities = np.zeros(ARM_DOF, dtype=np.float64)
        self.last_state = state
        self.last_state_at = time.monotonic()
        return True

    def wait_for_state(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            if self.poll():
                return
            time.sleep(0.01)
        raise ArmClientError(
            "No UR5 state within {:.1f} s. Is ./impedance_controller {} running, "
            "and is the robot in Remote Control?".format(timeout, self.config_path.name)
        )

    def age(self) -> float:
        if self.last_state_at is None:
            return float("inf")
        return time.monotonic() - self.last_state_at

    def require_fresh_state(self) -> np.ndarray:
        self.poll()
        if self.positions is None:
            raise ArmClientError("No UR5 state has been received.")
        age = self.age()
        if age > self.state_timeout:
            raise ArmClientError(
                "UR5 state is stale by {:.3f} s (limit {:.3f} s).".format(
                    age, self.state_timeout
                )
            )
        return self.positions.copy()

    # -- sending ---------------------------------------------------------
    def start_streaming(self) -> None:
        if not self._streaming:
            self._streamer.start()
            self._streaming = True

    def set_target(self, target_q: np.ndarray) -> None:
        if not self._streaming:
            raise ArmClientError("start_streaming() must be called before set_target()")
        self._streamer.set_target(target_q)

    def send_trajectory(self, times: np.ndarray, path: np.ndarray) -> None:
        """Hand the controller a timed joint path; it interpolates internally."""
        times = np.asarray(times, dtype=np.float64)
        path = np.asarray(path, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != ARM_DOF:
            raise ValueError("path must have shape (N, 6)")
        if times.shape != (path.shape[0],):
            raise ValueError("times must have one entry per path row")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(path)):
            raise ValueError("Trajectory contains non-finite values")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing")
        self.command_socket.send_json(
            {"time": times.tolist(), "path": path.tolist()}
        )

    def brake(
        self,
        duration_s: float = DEFAULT_BRAKE_SECONDS,
        frequency_hz: float = DEFAULT_BRAKE_HZ,
    ) -> bool:
        """Request ~zero joint velocity by re-commanding the measured position."""
        if self._streaming:
            self._streamer.clear_target()
        latest = self.positions.copy() if self.positions is not None else None
        deadline = time.monotonic() + float(duration_s)
        period = 1.0 / float(frequency_hz)
        sent = 0
        while time.monotonic() < deadline:
            if self.poll() and self.positions is not None:
                latest = self.positions.copy()
            if latest is not None:
                self.command_socket.send_json({"target_q": latest.tolist()})
                sent += 1
            time.sleep(period)
        return sent > 0

    def close(self) -> None:
        if self._streaming:
            self._streamer.stop()
            self._streaming = False
        self.command_socket.close(linger=0)
        self.state_socket.close(linger=0)
        self.context.term()

    def __enter__(self) -> "ArmClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
