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
