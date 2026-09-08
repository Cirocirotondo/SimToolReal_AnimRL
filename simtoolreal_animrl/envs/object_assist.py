"""Annealed object-assist wrench: a PD controller plus gravity compensation.

Grasping is the hard part of this task, so the object is helped toward the
demonstrated pose by an external wrench applied at the cube's centre of mass.
The wrench is multiplied by a scale that decays to zero over a configured
iteration window, which leaves the policy learning the *residual* force the
hand must supply. Once the scale reaches zero the environment is physically
identical to the unassisted one.

The functions here are pure tensor maths so they can be tested without Isaac
Gym; the environment owns the buffers and the actual force application.
"""

import math
from typing import NamedTuple, Tuple

import torch


class ObjectAssistSettings(NamedTuple):
    """Validated ``cfg.object_assist`` values."""

    enabled: bool
    schedule: str
    start_iteration: int
    end_iteration: int
    initial_scale: float
    final_scale: float
    position_stiffness_n_per_m: float
    position_damping_ns_per_m: float
    orientation_stiffness_nm_per_rad: float
    orientation_damping_nms_per_rad: float
    gravity_compensation: bool
    torque_enabled: bool
    max_force_n: float
    max_torque_nm: float
    active_from_reference_index: int


def resolve_object_assist_settings(
    assist_cfg, reference_last_index: int
) -> ObjectAssistSettings:
    """Validate the assist configuration section and return its settings."""
    schedule = str(assist_cfg.schedule)
    if schedule not in ("linear", "constant"):
        raise ValueError(
            "Unsupported object_assist.schedule {!r}".format(schedule)
        )
    start_iteration = int(assist_cfg.start_iteration)
    end_iteration = int(assist_cfg.end_iteration)
    if start_iteration < 0:
        raise ValueError("object_assist.start_iteration cannot be negative")
    if end_iteration < start_iteration:
        raise ValueError(
            "object_assist.end_iteration must not precede start_iteration"
        )
    if schedule == "linear" and end_iteration == start_iteration:
        raise ValueError(
            "A linear object-assist schedule needs end_iteration > "
            "start_iteration"
        )
    initial_scale = float(assist_cfg.initial_scale)
    final_scale = float(assist_cfg.final_scale)
    for name, value in (
        ("initial_scale", initial_scale),
        ("final_scale", final_scale),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "object_assist.{} must be finite and non-negative".format(name)
            )
    gains = {
        "position_stiffness_n_per_m": float(
            assist_cfg.position_stiffness_n_per_m
        ),
        "position_damping_ns_per_m": float(assist_cfg.position_damping_ns_per_m),
        "orientation_stiffness_nm_per_rad": float(
            assist_cfg.orientation_stiffness_nm_per_rad
        ),
        "orientation_damping_nms_per_rad": float(
            assist_cfg.orientation_damping_nms_per_rad
        ),
    }
    for name, value in gains.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "object_assist.{} must be finite and non-negative".format(name)
            )
    limits = {
        "max_force_n": float(assist_cfg.max_force_n),
        "max_torque_nm": float(assist_cfg.max_torque_nm),
    }
    for name, value in limits.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                "object_assist.{} must be finite and positive".format(name)
            )
    active_from = int(assist_cfg.active_from_reference_index)
    if not 0 <= active_from <= int(reference_last_index):
        raise ValueError(
            "object_assist.active_from_reference_index must lie in [0, {}]".format(
                int(reference_last_index)
            )
        )
    return ObjectAssistSettings(
        enabled=bool(assist_cfg.enabled),
        schedule=schedule,
        start_iteration=start_iteration,
        end_iteration=end_iteration,
        initial_scale=initial_scale,
        final_scale=final_scale,
        gravity_compensation=bool(assist_cfg.gravity_compensation),
        torque_enabled=bool(assist_cfg.torque_enabled),
        active_from_reference_index=active_from,
        **gains,
        **limits,
    )


def assist_scale_at(settings: ObjectAssistSettings, iteration: int) -> float:
    """Return the assist scale for a PPO iteration.

    ``constant`` holds ``initial_scale`` forever. ``linear`` holds it until
    ``start_iteration``, interpolates down to ``final_scale`` at
    ``end_iteration``, and stays there afterwards, so a run that trains past
    the window is exactly the unassisted problem.
    """
    if not settings.enabled:
        return 0.0
    if settings.schedule == "constant":
        return settings.initial_scale
    iteration = int(iteration)
    if iteration <= settings.start_iteration:
        return settings.initial_scale
    if iteration >= settings.end_iteration:
        return settings.final_scale
    progress = (iteration - settings.start_iteration) / float(
        settings.end_iteration - settings.start_iteration
    )
    return settings.initial_scale + progress * (
        settings.final_scale - settings.initial_scale
    )


def _clamp_vector_norm(vectors: torch.Tensor, maximum: float) -> torch.Tensor:
    """Scale rows down to ``maximum`` length, preserving their direction."""
    norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    factor = (maximum / norms.clamp_min(1.0e-9)).clamp(max=1.0)
    return vectors * factor


def orientation_error_rotation_vector(
    orientation: torch.Tensor, reference_orientation: torch.Tensor
) -> torch.Tensor:
    """Return the world-frame rotation vector taking ``orientation`` to the reference.

    Both inputs are xyzw quaternions. The result is ``axis * angle`` for the
    shortest rotation, so its norm is the same geodesic angle the object
    orientation reward uses.
    """
    conjugate = torch.cat(
        (-orientation[..., :3], orientation[..., 3:4]), dim=-1
    )
    left_xyz, left_w = reference_orientation[..., :3], reference_orientation[..., 3:4]
    right_xyz, right_w = conjugate[..., :3], conjugate[..., 3:4]
    error_xyz = (
        left_w * right_xyz
        + right_w * left_xyz
        + torch.cross(left_xyz, right_xyz, dim=-1)
    )
    error_w = left_w * right_w - (left_xyz * right_xyz).sum(dim=-1, keepdim=True)
    error = torch.nn.functional.normalize(
        torch.cat((error_xyz, error_w), dim=-1), dim=-1
    )
    # q and -q are the same rotation; the positive-w representative is the one
    # whose angle is the shortest path rather than its 2*pi complement.
    error = torch.where(error[..., 3:4] < 0.0, -error, error)
    sin_half_angle = torch.linalg.vector_norm(error[..., :3], dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half_angle, error[..., 3:4].clamp(-1.0, 1.0))
    return error[..., :3] * (angle / sin_half_angle.clamp_min(1.0e-9))


def object_reward_gate(
    assist_enabled: bool,
    gate_enabled: bool,
    assist_scale: float,
) -> float:
    """Share of the object reward the policy has to earn for itself.

    One while the assist is off, absent, or the gate is disabled for an
    ablation; falling to zero at full assist. A pinned cube tracks its
    demonstrated pose whatever the hand does, so paying the object terms in
    full then rewards the wrench rather than the policy, and nothing pushes it
    to take the load over before the assist anneals away.
    """
    if not (assist_enabled and gate_enabled):
        return 1.0
    return max(0.0, 1.0 - float(assist_scale))


def object_assist_wrench(
    position: torch.Tensor,
    orientation: torch.Tensor,
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    reference_root_state: torch.Tensor,
    settings: ObjectAssistSettings,
    scale: float,
    mass_kg: float,
    gravity: torch.Tensor,
    active: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the world-frame ``(force, torque)`` applied at the object's centre.

    ``reference_root_state`` is the demonstrated 13-value root state of the
    object in world coordinates, matching the layout Isaac Gym uses. The
    gravity-compensation term is independent of the pose error, so at scale 1
    a stationary object at its target floats instead of falling.
    """
    if reference_root_state.shape != (position.shape[0], 13):
        raise ValueError("Expected one 13-value reference root state per environment")
    if active.shape != position.shape[:1]:
        raise ValueError("Expected one assist activation flag per environment")
    scale = float(scale)
    gate = (active.to(dtype=position.dtype) * scale).unsqueeze(-1)

    position_error = reference_root_state[:, 0:3] - position
    linear_velocity_error = reference_root_state[:, 7:10] - linear_velocity
    force = (
        settings.position_stiffness_n_per_m * position_error
        + settings.position_damping_ns_per_m * linear_velocity_error
    )
    if settings.gravity_compensation:
        force = force - float(mass_kg) * gravity
    force = _clamp_vector_norm(force, settings.max_force_n) * gate

    if settings.torque_enabled:
        rotation_error = orientation_error_rotation_vector(
            orientation, reference_root_state[:, 3:7]
        )
        angular_velocity_error = reference_root_state[:, 10:13] - angular_velocity
        torque = (
            settings.orientation_stiffness_nm_per_rad * rotation_error
            + settings.orientation_damping_nms_per_rad * angular_velocity_error
        )
        torque = _clamp_vector_norm(torque, settings.max_torque_nm) * gate
    else:
        torque = torch.zeros_like(force)
    return force, torque
