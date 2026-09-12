"""Policy-side peer of ``dg5f_policy_ros_bridge.py``.

The bridge is reused unmodified from the SimToolReal deployment tree. It runs
under the ROS 2 Humble interpreter (Python 3.10) while the policy runs under
the AnimRL interpreter (Python 3.11); the two exchange only the latest hand
state and target as JSON over localhost UDP.

Wire protocol (bridge -> policy, on ``state_port``)::

    {"type": "hand_state", "sequence": int,
     "positions": [20], "velocities": [20], "currents_ma": [20]?}

and (policy -> bridge, on ``command_port``)::

    {"type": "hand_target", "positions": [20]}

The bridge owns the last line of hand safety: it clips to the right-hand joint
limits, clamps each command's step, rejects targets while ``/joint_states`` is
stale, and holds the measured position when the policy stops sending. Nothing
here weakens that; the checks below only fail earlier and more loudly.
"""

from __future__ import annotations

import json
import socket
import time
from typing import Optional

import numpy as np

HAND_DOF = 20
DEFAULT_STATE_PORT = 5563
DEFAULT_COMMAND_PORT = 5562
DEFAULT_COMMAND_ADDRESS = "127.0.0.1"
DEFAULT_BIND_ADDRESS = "127.0.0.1"


class HandClientError(RuntimeError):
    """The hand state stream is absent, stale or malformed."""


class HandClient:
    def __init__(
        self,
        *,
        bind_address: str = DEFAULT_BIND_ADDRESS,
        state_port: int = DEFAULT_STATE_PORT,
        command_address: str = DEFAULT_COMMAND_ADDRESS,
        command_port: int = DEFAULT_COMMAND_PORT,
        state_timeout: float = 0.25,
    ) -> None:
        if state_timeout <= 0.0:
            raise ValueError("state_timeout must be positive")
        self.state_timeout = float(state_timeout)
        self.command_endpoint = (command_address, int(command_port))

        self.state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_socket.setblocking(False)
        self.state_socket.bind((bind_address, int(state_port)))
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.positions: Optional[np.ndarray] = None
        self.velocities: Optional[np.ndarray] = None
        self.currents_ma: Optional[np.ndarray] = None
        self.sequence: Optional[int] = None
        self.last_state_at: Optional[float] = None
        self.dropped_messages = 0

    # -- receiving -------------------------------------------------------
    def poll(self) -> bool:
        """Drain the socket, keeping only the newest valid state. True if updated."""
        updated = False
        while True:
            try:
                payload, _ = self.state_socket.recvfrom(65535)
            except BlockingIOError:
                break
            try:
                message = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.dropped_messages += 1
                continue
            if message.get("type") != "hand_state":
                self.dropped_messages += 1
                continue
            positions = np.asarray(message.get("positions", []), dtype=np.float64)
            velocities = np.asarray(message.get("velocities", []), dtype=np.float64)
            if positions.shape != (HAND_DOF,) or velocities.shape != (HAND_DOF,):
                self.dropped_messages += 1
                continue
            if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
                self.dropped_messages += 1
                continue
            self.positions = positions
            self.velocities = velocities
            currents = message.get("currents_ma")
            if currents is not None:
                current_array = np.asarray(currents, dtype=np.float64)
                if current_array.shape == (HAND_DOF,) and np.all(
                    np.isfinite(current_array)
                ):
                    self.currents_ma = current_array
            sequence = message.get("sequence")
            self.sequence = int(sequence) if isinstance(sequence, int) else None
            self.last_state_at = time.monotonic()
            updated = True
        return updated

    def wait_for_state(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            self.poll()
            if self.positions is not None:
                return
            time.sleep(0.01)
        raise HandClientError(
            "No hand state within {:.1f} s. Is dg5f_policy_ros_bridge.py running, "
            "and is the DG5F driver publishing joint states?".format(timeout)
        )

    def age(self) -> float:
        if self.last_state_at is None:
            return float("inf")
        return time.monotonic() - self.last_state_at

    def require_fresh_state(self) -> np.ndarray:
        self.poll()
        if self.positions is None:
            raise HandClientError("No hand state has been received.")
        age = self.age()
        if age > self.state_timeout:
            raise HandClientError(
                "Hand state is stale by {:.3f} s (limit {:.3f} s).".format(
                    age, self.state_timeout
                )
            )
        return self.positions.copy()

    def over_current_joints(self, threshold_ma: float) -> np.ndarray:
        if self.currents_ma is None:
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(np.abs(self.currents_ma) > float(threshold_ma))

    # -- sending ---------------------------------------------------------
    def send_target(self, positions: np.ndarray) -> None:
        target = np.asarray(positions, dtype=np.float64)
        if target.shape != (HAND_DOF,):
            raise ValueError("Hand target must have shape (20,)")
        if not np.all(np.isfinite(target)):
            raise ValueError("Hand target contains non-finite values")
        payload = {"type": "hand_target", "positions": target.tolist()}
        self.command_socket.sendto(
            json.dumps(payload, separators=(",", ":")).encode(),
            self.command_endpoint,
        )

    def hold_measured(self, duration_s: float = 0.5, frequency_hz: float = 100.0) -> bool:
        """Stream the measured position so the bridge brakes instead of drifting."""
        deadline = time.monotonic() + float(duration_s)
        period = 1.0 / float(frequency_hz)
        sent = 0
        while time.monotonic() < deadline:
            self.poll()
            if self.positions is not None:
                self.send_target(self.positions)
                sent += 1
            time.sleep(period)
        return sent > 0

    def close(self) -> None:
        self.state_socket.close()
        self.command_socket.close()

    def __enter__(self) -> "HandClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
