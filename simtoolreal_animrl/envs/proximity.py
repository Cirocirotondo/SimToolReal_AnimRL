"""Pure fingertip-to-cuboid proximity shaping helpers."""

import torch


def fingertip_cuboid_proximity(
    fingertip_positions_cube: torch.Tensor,
    cuboid_half_extents: torch.Tensor,
    std_m: float,
    active: torch.Tensor,
):
    """Return gated reward, mean surface distance, and the per-finger distances.

    ``fingertip_positions_cube`` contains point positions in the cuboid frame.
    The Euclidean point-to-box distance is zero on or inside the cuboid and is
    exact at faces, edges and corners. Each selected fingertip gets its own
    Gaussian before the values are averaged, so all selected fingers are
    encouraged to approach rather than only the closest one.

    The third return value keeps every selected finger separate: the mean hides
    whether the hand closed evenly or one finger reached the cube alone.
    """
    if (
        fingertip_positions_cube.ndim != 3
        or fingertip_positions_cube.shape[-1] != 3
    ):
        raise ValueError("Expected fingertip positions with shape (N, F, 3)")
    if cuboid_half_extents.shape != (3,):
        raise ValueError("Expected cuboid half extents with shape (3,)")
    if active.shape != fingertip_positions_cube.shape[:1]:
        raise ValueError("Expected one proximity activation flag per environment")
    std_m = float(std_m)
    if std_m <= 0.0:
        raise ValueError("Proximity standard deviation must be positive")

    outside = (
        fingertip_positions_cube.abs() - cuboid_half_extents.view(1, 1, 3)
    ).clamp_min(0.0)
    distances_m = torch.linalg.vector_norm(outside, dim=2)
    per_finger_reward = torch.exp(-distances_m.square() / (2.0 * std_m**2))
    reward = per_finger_reward.mean(dim=1) * active.to(
        dtype=per_finger_reward.dtype
    )
    return reward, distances_m.mean(dim=1), distances_m
