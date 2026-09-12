"""Exact 112-D observation and residual-action contracts used by AnimRL."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .constants import (
    ACTION_DIM,
    BASE_OBSERVATION_DIM,
    FINGERTIP_OFFSETS,
    PALM_ORIENTATION_IN_WRIST_XYZW,
    PALM_POSITION_IN_WRIST,
)


def normalize_canonical_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Quaternion must be finite and non-zero")
    normalized = quaternion / norm
    return -normalized if normalized[3] < 0.0 else normalized


def quat_conjugate_xyzw(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    return np.concatenate((-quaternion[:3], quaternion[3:4]))


def quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_xyz, left_w = left[:3], left[3]
    right_xyz, right_w = right[:3], right[3]
    xyz = (
        left_w * right_xyz
        + right_w * left_xyz
        + np.cross(left_xyz, right_xyz)
    )
    w = left_w * right_w - float(np.dot(left_xyz, right_xyz))
    return np.concatenate((xyz, np.asarray([w], dtype=np.float64)))


def quat_rotate_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quaternion = normalize_canonical_quaternion(quaternion)
    vector = np.asarray(vector, dtype=np.float64)
    q_xyz = quaternion[:3]
    uv = np.cross(q_xyz, vector)
    uuv = np.cross(q_xyz, uv)
    return vector + 2.0 * (quaternion[3] * uv + uuv)


def quat_rotate_inverse_xyzw(
    quaternion: np.ndarray, vector: np.ndarray
) -> np.ndarray:
    return quat_rotate_xyzw(quat_conjugate_xyzw(quaternion), vector)


def quaternion_to_rotation_6d(quaternion: np.ndarray) -> np.ndarray:
    """The first two columns of the rotation matrix, flattened to six values.

    Mirrors ``_quaternion_to_rotation_6d`` in the Isaac Gym environment. It
    must stay a bit-for-bit equivalent contract: the policy deployed here is
    the one trained there, and a rotation encoded differently is simply a
    different observation. Note this deliberately does NOT canonicalize -- q
    and -q give the same matrix, which is the entire point of the encoding.
    """
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Quaternion must be finite and non-zero")
    quaternion = quaternion / norm
    x_axis = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    y_axis = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    q_xyz, q_w = quaternion[:3], quaternion[3]

    def rotate(vector):
        uv = np.cross(q_xyz, vector)
        uuv = np.cross(q_xyz, uv)
        return vector + 2.0 * (q_w * uv + uuv)

    return np.concatenate((rotate(x_axis), rotate(y_axis)))


def palm_and_fingertips_from_body_state(
    wrist_position_world: np.ndarray,
    wrist_orientation_world_xyzw: np.ndarray,
    fingertip_body_positions_world: np.ndarray,
    fingertip_body_orientations_world_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wrist_orientation = normalize_canonical_quaternion(
        wrist_orientation_world_xyzw
    )
    palm_position_world = np.asarray(wrist_position_world, dtype=np.float64) + (
        quat_rotate_xyzw(wrist_orientation, PALM_POSITION_IN_WRIST)
    )
    palm_orientation_world = normalize_canonical_quaternion(
        quat_multiply_xyzw(
            wrist_orientation, PALM_ORIENTATION_IN_WRIST_XYZW
        )
    )

    body_positions = np.asarray(fingertip_body_positions_world, dtype=np.float64)
    body_orientations = np.asarray(
        fingertip_body_orientations_world_xyzw, dtype=np.float64
    )
    if body_positions.shape != (5, 3) or body_orientations.shape != (5, 4):
        raise ValueError("Expected five fingertip body poses")
    tip_positions = np.empty((5, 3), dtype=np.float64)
    for index in range(5):
        tip_positions[index] = body_positions[index] + quat_rotate_xyzw(
            body_orientations[index], FINGERTIP_OFFSETS[index]
        )
    return palm_position_world, palm_orientation_world, tip_positions


def normalize_joint_positions(
    positions: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    lower = np.asarray(lower_limits, dtype=np.float64)
    upper = np.asarray(upper_limits, dtype=np.float64)
    if positions.shape != (ACTION_DIM,) or lower.shape != positions.shape or upper.shape != positions.shape:
        raise ValueError("Joint positions and limits must all have shape (26,)")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(lower >= upper):
        raise ValueError("Joint limits must be finite, non-empty intervals")
    return np.clip(2.0 * (positions - lower) / (upper - lower) - 1.0, -1.0, 1.0)


def build_observation(
    state: Mapping[str, np.ndarray],
    previous_targets: np.ndarray,
    phase: float,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> np.ndarray:
    """Build the raw observation in the exact order used during training."""
    q = np.asarray(state["joint_positions"], dtype=np.float64)
    dq = np.asarray(state["joint_velocities"], dtype=np.float64)
    previous_targets = np.asarray(previous_targets, dtype=np.float64)
    if q.shape != (ACTION_DIM,) or dq.shape != (ACTION_DIM,):
        raise ValueError("MuJoCo robot state must contain 26 joints")
    if previous_targets.shape != (ACTION_DIM,):
        raise ValueError("previous_targets must have shape (26,)")
    if not np.isfinite(phase):
        raise ValueError("phase must be finite")

    palm_position_world, palm_orientation_world, fingertip_positions_world = (
        palm_and_fingertips_from_body_state(
            state["wrist_position_world"],
            state["wrist_orientation_world_xyzw"],
            state["fingertip_body_positions_world"],
            state["fingertip_body_orientations_world_xyzw"],
        )
    )
    robot_position_world = np.asarray(state["robot_position_world"], dtype=np.float64)
    robot_orientation_world = normalize_canonical_quaternion(
        state["robot_orientation_world_xyzw"]
    )
    palm_position_robot = quat_rotate_inverse_xyzw(
        robot_orientation_world, palm_position_world - robot_position_world
    )
    palm_rotation_robot = quaternion_to_rotation_6d(
        quat_multiply_xyzw(
            quat_conjugate_xyzw(robot_orientation_world), palm_orientation_world
        )
    )

    fingertip_positions_palm = np.stack(
        [
            quat_rotate_inverse_xyzw(
                palm_orientation_world, position - palm_position_world
            )
            for position in fingertip_positions_world
        ]
    )
    cube_position_world = np.asarray(state["cube_position_world"], dtype=np.float64)
    cube_orientation_world = normalize_canonical_quaternion(
        state["cube_orientation_world_xyzw"]
    )
    cube_center_palm = quat_rotate_inverse_xyzw(
        palm_orientation_world, cube_position_world - palm_position_world
    )
    cube_rotation_palm = quaternion_to_rotation_6d(
        quat_multiply_xyzw(
            quat_conjugate_xyzw(palm_orientation_world), cube_orientation_world
        )
    )

    observation = np.concatenate(
        (
            normalize_joint_positions(q, lower_limits, upper_limits),
            previous_targets,
            dq,
            np.asarray([np.clip(phase, 0.0, 1.0)]),
            palm_position_robot,
            palm_rotation_robot,
            fingertip_positions_palm.reshape(-1),
            cube_rotation_palm,
            cube_center_palm,
        )
    ).astype(np.float32)
    if observation.shape != (BASE_OBSERVATION_DIM,):
        raise RuntimeError(
            "Observation has shape {}, expected ({},)".format(
                observation.shape, BASE_OBSERVATION_DIM
            )
        )
    if not np.all(np.isfinite(observation)):
        raise RuntimeError("Observation contains NaN or infinite values")
    return observation


def actions_to_position_targets(
    actions: np.ndarray,
    default_positions: np.ndarray,
    arm_scale: float,
    hand_scale: float,
    residual_clip: float,
) -> np.ndarray:
    """Apply AnimRL's unbounded residual-action mapping."""
    actions = np.asarray(actions, dtype=np.float64)
    defaults = np.asarray(default_positions, dtype=np.float64)
    if actions.shape != (ACTION_DIM,) or defaults.shape != (ACTION_DIM,):
        raise ValueError("Actions and default_positions must have shape (26,)")
    scales = np.concatenate(
        (
            np.full(6, float(arm_scale), dtype=np.float64),
            np.full(20, float(hand_scale), dtype=np.float64),
        )
    )
    residual = np.clip(
        actions * scales, -float(residual_clip), float(residual_clip)
    )
    return (defaults + residual).astype(np.float64)
