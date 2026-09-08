"""Robot and observation constants shared by the MuJoCo sim2sim backend."""

import numpy as np


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

WRIST_BODY_NAME = "wrist_3_link"
FINGERTIP_BODY_NAMES = tuple("rl_dg_{}_4".format(finger) for finger in range(1, 6))
FINGERTIP_OFFSETS = np.asarray(
    (
        (0.0, 0.0363, 0.0),
        (0.0, 0.0, 0.0255),
        (0.0, 0.0, 0.0255),
        (0.0, 0.0, 0.0255),
        (0.0, 0.0, 0.0363),
    ),
    dtype=np.float64,
)

# The fixed wrist -> 60-degree mount -> palm chain is collapsed by Isaac Gym.
# These values reproduce the palm frame used during training.
PALM_POSITION_IN_WRIST = np.asarray((0.0, 0.0, 0.0738), dtype=np.float64)
PALM_ORIENTATION_IN_WRIST_XYZW = np.asarray(
    (0.0, 0.0, 0.5, 0.8660254037844386), dtype=np.float64
)

BASE_OBSERVATION_DIM = 108
ACTION_DIM = 26
