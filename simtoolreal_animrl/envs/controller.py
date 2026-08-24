"""UR5e + DG5F joint layout and corrected low-level PD configuration."""

from typing import Sequence, Tuple

import numpy as np
from isaacgym import gymapi


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

# Source of truth copied from the corrected hand_dofs == 20 branch of
# simtoolreal/isaacgymenvs/tasks/simtoolreal/utils.py and from the verified
# Isaac Gym demonstration viewer.
HAND_PD_STIFFNESS = (
    42.9718, 400.0, 42.9718, 42.9718,
    42.9718, 42.9718, 42.9718, 42.9718,
    42.9718, 42.9718, 42.9718, 42.9718,
    42.9718, 42.9718, 42.9718, 42.9718,
    42.9718, 42.9718, 42.9718, 42.9718,
)
HAND_PD_DAMPING = (
    0.1, 0.9475, 0.3012, 0.1821,
    0.7523, 0.4126, 0.2856, 0.1365,
    0.7587, 0.4126, 0.2856, 0.1365,
    0.7274, 0.4126, 0.2856, 0.1365,
    0.2662, 0.4796, 0.3012, 0.1821,
)

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


def configure_pd_properties(gym, asset, demo_to_asset: np.ndarray):
    properties = gym.get_asset_dof_properties(asset)
    properties["driveMode"].fill(int(gymapi.DOF_MODE_POS))
    hand_indices = demo_to_asset[len(ARM_JOINT_NAMES):]
    properties["stiffness"][hand_indices] = HAND_PD_STIFFNESS
    properties["damping"][hand_indices] = HAND_PD_DAMPING
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
