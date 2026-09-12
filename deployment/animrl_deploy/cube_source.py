"""Where the observed cube pose comes from.

The 108-D observation ends with the cube's rotation and centre expressed in the
palm frame, so *something* must supply a cube pose every control step. Three
sources exist, in increasing order of risk:

``DemonstrationCube``
    Replays the cube trajectory recorded in the demonstration ``.npz``. Nothing
    is measured; the cube pose is a function of the reference index alone. This
    is the hardware-in-the-loop source: it lets the arm and hand be commissioned
    separately, with a repeatable observation, before any camera is trusted.

``FrozenCube``
    Holds one demonstration sample forever. Useful to check that a stationary
    observation produces a stationary action.

``PoseEstimationCube``
    Subscribes to the tag pose estimator. This is the only source that closes
    the loop on the real cube, and the only one whose frame convention has to be
    verified before use -- see the warning on that class.

All three return a pose in the *demonstration's* frame convention, because that
is what ``HardwareKinematics.update`` maps into the MuJoCo world using the same
transform sim2sim applies at reset.
"""

from __future__ import annotations

import time
from typing import Optional, Protocol, Tuple

import numpy as np
import zmq

DEFAULT_POSE_ADDRESS = "tcp://127.0.0.1:5557"

CubeState = Tuple[np.ndarray, np.ndarray, np.ndarray]


class CubeSourceError(RuntimeError):
    """The cube pose stream is absent, stale or malformed."""


class CubeSource(Protocol):
    def cube_state(self, reference_index: int) -> CubeState:
        """Return (pose_xyzw[7], linear_velocity[3], angular_velocity[3])."""


class DemonstrationCube:
    """Cube pose replayed from the demonstration, indexed by reference sample."""

    name = "demonstration"

    def __init__(self, reference) -> None:
        self.reference = reference

    def cube_state(self, reference_index: int) -> CubeState:
        import torch

        index = int(np.clip(reference_index, 0, self.reference.last_index))
        sample = self.reference.sample(torch.tensor([index], dtype=torch.long))
        return (
            sample.cube_pose[0].numpy().astype(np.float64),
            sample.cube_linear_velocity[0].numpy().astype(np.float64),
            sample.cube_angular_velocity[0].numpy().astype(np.float64),
        )


class FrozenCube:
    """One demonstration sample, held for the whole rollout."""

    name = "frozen"

    def __init__(self, reference, reference_index: int) -> None:
        import torch

        index = int(np.clip(reference_index, 0, reference.last_index))
        sample = reference.sample(torch.tensor([index], dtype=torch.long))
        self._pose = sample.cube_pose[0].numpy().astype(np.float64)
        self._zero = np.zeros(3, dtype=np.float64)

    def cube_state(self, reference_index: int) -> CubeState:
        return self._pose.copy(), self._zero.copy(), self._zero.copy()


class PoseEstimationCube:
    """Live cube pose from the tag pose estimator.

    WARNING -- frame convention. The estimator is run with a ``*_robot_frame``
    config and publishes a position plus a rotation matrix. This class assumes
    that frame is the one the demonstration recorded its ``cube_pose`` in, since
    the demonstration was captured through the same estimator. That assumption
    is plausible but NOT verified by this code, and a wrong frame is silently
    wrong: the policy would receive a mirrored or rotated cube and reach for the
    wrong place. Before ever enabling this source, park the real cube at a known
    demonstration sample and compare ``cube_state()`` against
    ``DemonstrationCube.cube_state()`` for that index. ``--check-cube-frame``
    in ``run_policy_real.py`` performs exactly that comparison.

    Velocities are reported as zero: the estimator publishes pose only, and a
    finite difference of a noisy tag pose is worse than a zero. The trained
    observation does not read cube velocity, so this costs nothing.
    """

    name = "pose-estimation"

    def __init__(
        self,
        address: str = DEFAULT_POSE_ADDRESS,
        *,
        board_id: str = "0",
        minimum_confidence: float = 0.0,
        pose_timeout: float = 0.5,
        context: Optional[zmq.Context] = None,
    ) -> None:
        self._owns_context = context is None
        self.context = context or zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.connect(address)
        self.address = address
        self.board_id = str(board_id)
        self.minimum_confidence = float(minimum_confidence)
        self.pose_timeout = float(pose_timeout)
        self._pose: Optional[np.ndarray] = None
        self._last_pose_at: Optional[float] = None
        self._zero = np.zeros(3, dtype=np.float64)

    @staticmethod
    def _rotation_matrix_to_xyzw(rotation: np.ndarray) -> np.ndarray:
        # Project numerical drift onto the closest proper rotation first.
        u, _, vt = np.linalg.svd(rotation)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vt
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * scale
            x = (rotation[2, 1] - rotation[1, 2]) / scale
            y = (rotation[0, 2] - rotation[2, 0]) / scale
            z = (rotation[1, 0] - rotation[0, 1]) / scale
        elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif rotation[1, 1] > rotation[2, 2]:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
        quaternion = np.asarray((x, y, z, w), dtype=np.float64)
        return quaternion / np.linalg.norm(quaternion)

    def poll(self) -> bool:
        updated = False
        while True:
            try:
                message = self.socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            poses = message.get("poses")
            if not isinstance(poses, dict):
                continue
            pose = poses.get(self.board_id)
            if not isinstance(pose, dict):
                continue
            if float(pose.get("confidence", 0.0)) < self.minimum_confidence:
                continue
            position = np.asarray(pose.get("position"), dtype=np.float64)
            rotation = np.asarray(pose.get("rotation_matrix"), dtype=np.float64)
            if position.shape != (3,) or rotation.shape != (3, 3):
                continue
            if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
                continue
            self._pose = np.concatenate(
                (position, self._rotation_matrix_to_xyzw(rotation))
            )
            self._last_pose_at = time.monotonic()
            updated = True
        return updated

    def wait_for_pose(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            self.poll()
            if self._pose is not None:
                return
            time.sleep(0.01)
        raise CubeSourceError(
            "No cube pose on {} for board '{}' within {:.1f} s. Is "
            "run_pose_estimation.py running?".format(
                self.address, self.board_id, timeout
            )
        )

    def cube_state(self, reference_index: int) -> CubeState:
        self.poll()
        if self._pose is None:
            raise CubeSourceError("No cube pose has been received.")
        age = time.monotonic() - float(self._last_pose_at)
        if age > self.pose_timeout:
            raise CubeSourceError(
                "Cube pose is stale by {:.3f} s (limit {:.3f} s).".format(
                    age, self.pose_timeout
                )
            )
        return self._pose.copy(), self._zero.copy(), self._zero.copy()

    def close(self) -> None:
        self.socket.close(linger=0)
        if self._owns_context:
            self.context.term()
