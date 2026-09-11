"""Perturb the reference pose a reset starts from.

Reference state initialisation places the robot exactly on a demonstration
frame. The real arm cannot be placed exactly on anything, so a policy trained
only on exact frames has never seen the states it will actually be started from,
and its first corrections are made from a distribution it has not visited.

It also blurs an artefact this project measured: hand action rate jumps 54x at
reference frame 690, before any contact (correlation with contact 0.08), which
is where the RSI distribution's mass begins. That is a seam between a
barely-trained approach and a heavily-trained grasp. Noisy starts spread mass
across the boundary rather than stacking it on one side.

Kept free of isaacgym so the clamping logic is testable without a simulation.
"""

import torch


def perturb_reference_pose(
    positions,
    velocities,
    arm_joint_count,
    position_noise_arm_rad,
    position_noise_hand_rad,
    velocity_noise_scale,
    lower_limits=None,
    upper_limits=None,
    generator=None,
):
    """Return ``(positions, velocities)`` with uniform noise added.

    Arm and hand take separate magnitudes: the arm is a position-controlled
    industrial manipulator that lands within a milliradian or so, while the
    tendon-driven hand is far less repeatable. Noise is uniform rather than
    Gaussian so the perturbation has a hard bound -- a tail sample that put a
    finger through the cube would inject a contact impulse at reset.

    Positions are clamped back inside the joint limits when they are given,
    because a reset outside them is a state the robot can never occupy.
    """
    arm_joint_count = int(arm_joint_count)
    positions = positions.clone()
    velocities = velocities.clone()

    def uniform(shape, scale):
        if scale <= 0.0:
            return None
        noise = torch.rand(
            shape, device=positions.device, dtype=positions.dtype,
            generator=generator,
        )
        return (noise * 2.0 - 1.0) * float(scale)

    arm_noise = uniform(
        positions[:, :arm_joint_count].shape, position_noise_arm_rad
    )
    if arm_noise is not None:
        positions[:, :arm_joint_count] += arm_noise
    hand_noise = uniform(
        positions[:, arm_joint_count:].shape, position_noise_hand_rad
    )
    if hand_noise is not None:
        positions[:, arm_joint_count:] += hand_noise

    if float(velocity_noise_scale) > 0.0:
        # Proportional to the reference velocity: a joint that is stationary in
        # the demonstration should not be given motion it never had.
        scale = float(velocity_noise_scale)
        noise = torch.rand(
            velocities.shape, device=velocities.device, dtype=velocities.dtype,
            generator=generator,
        )
        velocities += (noise * 2.0 - 1.0) * scale * velocities.abs()

    if lower_limits is not None and upper_limits is not None:
        positions = torch.clamp(positions, lower_limits, upper_limits)
    return positions, velocities
