"""UR5e + DG5F joint layout and corrected low-level PD configuration."""

from typing import Sequence, Tuple

import numpy as np
from isaacgym import gymapi

from simtoolreal_animrl.envs.pd_gains import (
    ARM_PD_DAMPING,
    ARM_PD_STIFFNESS,
    HAND_PD_DAMPING,
    HAND_PD_STIFFNESS,
    scale_gains,
)


ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
HAND_JOINT_NAMES = tuple(
    "rj_dg_{}_{}".format(finger, joint)
    for finger in range(1, 6)
    for joint in range(1, 5)
)
JOINT_NAMES = ARM_JOINT_NAMES + HAND_JOINT_NAMES

# Finite UR5e position-drive gains tuned at 60 Hz / two PhysX substeps against
WRIST_BODY_NAME = "wrist_3_link"
WRIST_COLLISION_HAND_BODY_NAMES = ("rl_dg_1_2", "rl_dg_4_2")


def validate_joint_order(gym, asset) -> np.ndarray:
    asset_names = tuple(gym.get_asset_dof_names(asset))
    missing = sorted(set(JOINT_NAMES) - set(asset_names))
    extra = sorted(set(asset_names) - set(JOINT_NAMES))
    if missing or extra or len(asset_names) != len(JOINT_NAMES):
        raise ValueError(
            "Robot DOFs do not match the demonstration; missing={}, extra={}".format(
                missing, extra
            )
        )
    return np.asarray([asset_names.index(name) for name in JOINT_NAMES], dtype=np.int64)


def configure_pd_properties(
    gym,
    asset,
    demo_to_asset: np.ndarray,
    arm_stiffness_scale: float = 1.0,
    arm_damping_scale: float = 1.0,
    hand_stiffness_scale: float = 1.0,
    hand_damping_scale: float = 1.0,
):
    """Install the position-drive gains, optionally softened per limb.

    Lowering a joint's stiffness lowers its closed-loop bandwidth
    ``omega = sqrt(k / J)``, so the drive filters the policy's step-to-step
    chatter instead of tracking it into the joint. It also *raises* the damping
    ratio ``zeta = d / (2 sqrt(k J))`` for a fixed damping, which is why a
    softer hand is both slower and better damped -- the combination that makes
    a policy safe to put on the real robot.
    """
    properties = gym.get_asset_dof_properties(asset)
    properties["driveMode"].fill(int(gymapi.DOF_MODE_POS))
    arm_indices = demo_to_asset[:len(ARM_JOINT_NAMES)]
    hand_indices = demo_to_asset[len(ARM_JOINT_NAMES):]
    properties["stiffness"][arm_indices] = scale_gains(
        ARM_PD_STIFFNESS, arm_stiffness_scale
    )
    properties["damping"][arm_indices] = scale_gains(
        ARM_PD_DAMPING, arm_damping_scale
    )
    properties["stiffness"][hand_indices] = scale_gains(
        HAND_PD_STIFFNESS, hand_stiffness_scale
    )
    properties["damping"][hand_indices] = scale_gains(
        HAND_PD_DAMPING, hand_damping_scale
    )
    return properties


def configure_asset_wrist_collision_filters(gym, asset) -> Tuple[int, ...]:
    """Filter only the two known wrist/hand mesh intersections on the asset."""
    body_names = tuple(gym.get_asset_rigid_body_names(asset))
    required = (WRIST_BODY_NAME,) + WRIST_COLLISION_HAND_BODY_NAMES
    missing = [name for name in required if name not in body_names]
    if missing:
        raise ValueError("Robot asset is missing collision bodies: {}".format(missing))

    shape_ranges = gym.get_asset_rigid_body_shape_indices(asset)
    shape_properties = gym.get_asset_rigid_shape_properties(asset)
    used_bits = 0
    for properties in shape_properties:
        used_bits |= int(properties.filter)

    allocated = []
    next_bit = 1
    for hand_body_name in WRIST_COLLISION_HAND_BODY_NAMES:
        while used_bits & next_bit:
            next_bit <<= 1
        if next_bit >= (1 << 31):
            raise RuntimeError("No collision-filter bit remains available")
        filter_bit = next_bit
        used_bits |= filter_bit
        allocated.append(filter_bit)

        for body_name in (WRIST_BODY_NAME, hand_body_name):
            shape_range = shape_ranges[body_names.index(body_name)]
            for shape_index in range(
                shape_range.start, shape_range.start + shape_range.count
            ):
                shape_properties[shape_index].filter |= filter_bit

    gym.set_asset_rigid_shape_properties(asset, shape_properties)
    return tuple(allocated)


def as_float32(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)
