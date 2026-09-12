"""The hand as nine points, measured from the bar rather than from the robot.

The demonstration is imitated in the cuboid's frame: where the palm is relative
to the bar, and where the five fingertips are relative to the bar. Move the bar
and the reference moves with it, which is the whole generalisation.

The palm contributes four points, not one. A pose is three numbers of position
and three of orientation, and mixing metres with radians in a reward means
inventing an exchange rate between them. Four rigidly attached points -- the
palm origin and three more at a lever arm along its axes -- carry the same six
degrees of freedom in one unit, so a single sigma in metres covers both. The
lever arm *is* the exchange rate, chosen rather than stumbled into: at
``L = 0.1`` m the term inherits the 0.05 m / 0.5 rad ratio the object reward
already uses. Up to that scale factor this is the 6D rotation representation.

Palm and fingertips keep separate Gaussians, for the same reason the arm and
hand joints do today: averaged into one term, five fingertips outvote four palm
points and the approach stops being paid for.

One property worth naming because a test asserts it. Expressed in the cuboid's
frame, the reference keypoints are **independent of where the cuboid is**: a
rigid transform of the whole scene cancels. So there is one reference curve for
every episode, not one per transform -- it is stored once, and if it ever varies
with the transform there is a frame bug somewhere upstream.
"""

import torch

from simtoolreal_animrl.envs.rotations import quat_rotate, quat_rotate_inverse


PALM_KEYPOINT_COUNT = 4
FINGERTIP_KEYPOINT_COUNT = 5
KEYPOINT_COUNT = PALM_KEYPOINT_COUNT + FINGERTIP_KEYPOINT_COUNT


def hand_keypoints(
    palm_position: torch.Tensor,
    palm_orientation: torch.Tensor,
    fingertip_positions: torch.Tensor,
    lever_arm_m: float,
) -> torch.Tensor:
    """Return ``(..., 9, 3)`` keypoints: palm origin, three axes, five fingertips.

    ``palm_position`` is ``(..., 3)``, ``palm_orientation`` a ``(..., 4)`` xyzw
    quaternion, ``fingertip_positions`` is ``(..., 5, 3)``. Everything must be in
    one frame; the caller decides which.
    """
    if fingertip_positions.shape[-2:] != (FINGERTIP_KEYPOINT_COUNT, 3):
        raise ValueError("Expected fingertip positions with shape (..., 5, 3)")
    lever_arm_m = float(lever_arm_m)
    if lever_arm_m <= 0.0:
        raise ValueError("The palm lever arm must be positive")

    batch = palm_position.shape[:-1]
    axes = torch.eye(
        3, dtype=palm_position.dtype, device=palm_position.device
    ) * lever_arm_m
    axes = axes.expand(*batch, 3, 3)
    orientation = palm_orientation.unsqueeze(-2).expand(*batch, 3, 4)
    axis_points = palm_position.unsqueeze(-2) + quat_rotate(orientation, axes)
    return torch.cat(
        (palm_position.unsqueeze(-2), axis_points, fingertip_positions), dim=-2
    )


def keypoints_in_object_frame(
    keypoints: torch.Tensor,
    object_position: torch.Tensor,
    object_orientation: torch.Tensor,
) -> torch.Tensor:
    """Express ``(..., K, 3)`` keypoints in the cuboid's frame.

    ``object_orientation`` should already be canonicalised -- see
    ``envs/cuboid_symmetry.py``. Feeding a raw pose estimate here is what makes
    a bar that never moved appear to jump.
    """
    count = keypoints.shape[-2]
    batch = keypoints.shape[:-2]
    relative = keypoints - object_position.unsqueeze(-2)
    orientation = object_orientation.unsqueeze(-2).expand(*batch, count, 4)
    return quat_rotate_inverse(orientation, relative)


def keypoint_tracking_error(
    keypoints_object_frame: torch.Tensor,
    reference_object_frame: torch.Tensor,
) -> torch.Tensor:
    """Mean squared keypoint distance, ``(..., K, 3)`` in, ``(...,)`` out.

    Squared distance per keypoint, then the mean over keypoints -- so the value
    is an RMS distance squared and a sigma in metres means what it reads as.
    """
    if keypoints_object_frame.shape != reference_object_frame.shape:
        raise ValueError(
            "Keypoints {} and reference {} must have the same shape".format(
                tuple(keypoints_object_frame.shape),
                tuple(reference_object_frame.shape),
            )
        )
    difference = keypoints_object_frame - reference_object_frame
    return difference.square().sum(dim=-1).mean(dim=-1)


def split_palm_and_fingertips(keypoints: torch.Tensor):
    """Split ``(..., 9, 3)`` into the palm block and the fingertip block."""
    if keypoints.shape[-2] != KEYPOINT_COUNT:
        raise ValueError("Expected {} keypoints".format(KEYPOINT_COUNT))
    return (
        keypoints[..., :PALM_KEYPOINT_COUNT, :],
        keypoints[..., PALM_KEYPOINT_COUNT:, :],
    )


def keypoint_gaussian(mean_squared_error: torch.Tensor, std_m: float) -> torch.Tensor:
    """The same Gaussian shaping every other tracking term in this project uses."""
    std_m = float(std_m)
    if std_m <= 0.0:
        raise ValueError("The keypoint standard deviation must be positive")
    return torch.exp(-mean_squared_error / (2.0 * std_m ** 2))
