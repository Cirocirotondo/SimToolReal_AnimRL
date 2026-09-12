"""MuJoCo driven as a pure forward-kinematics engine by measured hardware state.

The 108-D AnimRL observation needs palm and fingertip poses, which no encoder
reports. sim2sim reads them from its MuJoCo scene; on the robot we build the
*same* scene from the *same* saved config and write the measured joint angles
into it, then run ``mj_forward`` only.

Physics is never stepped here. ``AnimRLMujocoSim.step_for`` is the only method
that integrates, and this module does not call it. That is deliberate: the
robot integrates the physics, MuJoCo only answers "where is the palm, given
these 26 joint angles", using the identical kinematic chain, identical palm
offset, and identical joint limits that produced the training observation.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from simtoolreal_animrl.sim2sim.constants import ACTION_DIM
from simtoolreal_animrl.sim2sim.mujoco_sim import AnimRLMujocoSim, MujocoSceneConfig


class HardwareKinematics:
    """Forward kinematics for the measured robot, plus a cube pose to observe."""

    def __init__(
        self,
        run: Any,
        *,
        enable_viewer: bool = False,
        enable_reference_ghost: bool = False,
    ) -> None:
        self.config = MujocoSceneConfig.from_saved_config(
            run.repo_root,
            run.env_cfg,
            enable_viewer=bool(enable_viewer),
            enable_reference_ghost=bool(enable_reference_ghost),
        )
        self.sim = AnimRLMujocoSim(self.config)
        self._has_state = False

    # -- scene constants -------------------------------------------------
    @property
    def joint_lower_limits(self) -> np.ndarray:
        return self.sim.joint_lower_limits

    @property
    def joint_upper_limits(self) -> np.ndarray:
        return self.sim.joint_upper_limits

    # -- driving ---------------------------------------------------------
    def update(
        self,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        cube_pose_reference_frame: np.ndarray,
        cube_linear_velocity: np.ndarray,
        cube_angular_velocity: np.ndarray,
    ) -> Mapping[str, np.ndarray]:
        """Place the model at the measured state and return the observation state.

        ``cube_pose_reference_frame`` uses the demonstration's frame convention
        (position/quaternion as stored in the demo ``cube_pose``); it is mapped
        into the MuJoCo world by the same transform sim2sim uses at reset, so a
        demonstration cube pose and a policy observation stay consistent.
        """
        q = np.asarray(joint_positions, dtype=np.float64)
        dq = np.asarray(joint_velocities, dtype=np.float64)
        if q.shape != (ACTION_DIM,) or dq.shape != (ACTION_DIM,):
            raise ValueError("Measured robot state must have shape (26,)")
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)):
            raise ValueError("Measured robot state contains non-finite values")

        # reset() writes the joints, maps and writes the cube, and runs
        # mj_forward. It does not integrate.
        self.sim.reset(
            q,
            dq,
            np.asarray(cube_pose_reference_frame, dtype=np.float64),
            np.asarray(cube_linear_velocity, dtype=np.float64),
            np.asarray(cube_angular_velocity, dtype=np.float64),
        )
        self._has_state = True
        return self.sim.get_state()

    def state(self) -> Mapping[str, np.ndarray]:
        if not self._has_state:
            raise RuntimeError("update() must be called before state()")
        return self.sim.get_state()

    def set_reference_ghost(self, joint_positions: np.ndarray) -> None:
        self.sim.set_reference_ghost(np.asarray(joint_positions, dtype=np.float64))

    def sync_viewer(self) -> None:
        self.sim.sync_viewer()

    def viewer_is_running(self) -> bool:
        if self.sim.viewer is None:
            return True
        return self.sim.viewer_is_running()

    def close(self) -> None:
        self.sim.close()

    def __enter__(self) -> "HardwareKinematics":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
