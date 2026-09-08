"""Vectorized fingertip-contact diagnostics shared by training and tests."""

import torch


def fingertip_force_norms(
    net_contact_forces: torch.Tensor,
    fingertip_body_indices: torch.Tensor,
) -> torch.Tensor:
    """Per-fingertip net contact force magnitude, shape (num_envs, num_tips).

    Isaac Gym reports one net 3D contact-force vector per rigid body, so this
    aggregates every contact acting on that fingertip -- cube and table alike.
    It is not a per-pair fingertip-to-cube force.
    """
    fingertip_forces = net_contact_forces[:, fingertip_body_indices]
    return torch.linalg.vector_norm(fingertip_forces, dim=2)


def fingertip_force_observation_dim(contact_cfg) -> int:
    """Width the fingertip-force block adds to the observation vector.

    Zero unless the block is switched on, which is what keeps every existing
    experiment at its 108D observation and its checkpoints loadable.
    """
    if not bool(getattr(contact_cfg, "observe_fingertip_forces", False)):
        return 0
    return 3 * len(contact_cfg.fingertip_names)


def select_fingertip_forces(
    net_contact_forces: torch.Tensor,
    fingertip_body_indices: torch.Tensor,
) -> torch.Tensor:
    """Net contact force per selected fingertip, shape (num_envs, num_tips, 3).

    The vectors come out in the order the fingertips were configured, and carry
    the same caveat as `fingertip_force_norms`: Isaac Gym nets together every
    contact on that body, so a fingertip pressing the table reads like one
    pressing the cube.
    """
    return net_contact_forces[:, fingertip_body_indices]


def fingertip_force_observation(
    fingertip_forces: torch.Tensor,
    force_scale_n: float,
    clip: float,
) -> torch.Tensor:
    """Flatten fingertip force vectors into a scaled, clipped observation block.

    Takes the forces already rotated into the frame the observation uses, so
    this stays pure tensor arithmetic. Dividing by `force_scale_n` puts a firm
    grasp near unit scale, and the symmetric clip keeps a collision spike --
    worth several hundred newtons -- from swamping the inputs beside it.
    """
    scaled = fingertip_forces / float(force_scale_n)
    clip = float(clip)
    return scaled.clamp(-clip, clip).reshape(fingertip_forces.shape[0], -1)


def fingertip_contact_diagnostics(
    net_contact_forces: torch.Tensor,
    fingertip_body_indices: torch.Tensor,
    force_threshold_n: float,
):
    """Return per-env contact count, selected fraction, and mean force.

    A selected fingertip contributes exactly one binary contact when the norm
    of its net force vector exceeds the configured threshold, regardless of
    force size. The mean force averages over every selected fingertip, those
    below the threshold included, so it is not a mean contact force.
    """
    fingertip_force_n = fingertip_force_norms(
        net_contact_forces, fingertip_body_indices
    )
    contacts = fingertip_force_n > float(force_threshold_n)
    return (
        contacts.float().sum(dim=1),
        contacts.float().mean(dim=1),
        fingertip_force_n.mean(dim=1),
    )
