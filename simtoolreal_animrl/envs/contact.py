"""Vectorized fingertip-contact diagnostics shared by training and tests."""

import torch


def fingertip_contact_diagnostics(
    net_contact_forces: torch.Tensor,
    fingertip_body_indices: torch.Tensor,
    force_threshold_n: float,
):
    """Return per-env contact count, selected fraction, and mean force.

    Isaac Gym reports one net 3D contact-force vector per rigid body.  A
    selected fingertip contributes exactly one binary contact when the norm of
    that vector exceeds the configured threshold, regardless of force size.
    """
    fingertip_forces = net_contact_forces[:, fingertip_body_indices]
    fingertip_force_n = torch.linalg.vector_norm(fingertip_forces, dim=2)
    contacts = fingertip_force_n > float(force_threshold_n)
    return (
        contacts.float().sum(dim=1),
        contacts.float().mean(dim=1),
        fingertip_force_n.mean(dim=1),
    )
