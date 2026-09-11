"""Random external impulses, so the policy cannot assume an undisturbed world.

Domain randomisation over gains, friction and mass varies the world's
*parameters* but leaves it deterministic: nothing ever pushes the robot. A real
arm is knocked by cable drag, a real cube is nudged by an imperfect placement,
and a policy that has only ever been disturbed by its own actions has no
recovery behaviour at all.

Impulses are sampled per environment per step, sparse and short. Sparse because
a continuous push is a force field the policy learns to lean against, which is
the opposite of what this is for; short because an impulse is what a real
disturbance is -- a brief transfer of momentum, not a sustained load.
"""

import torch


def sample_impulses(
    num_envs,
    num_bodies,
    probability,
    magnitude_n,
    device,
    body_indices=None,
    generator=None,
):
    """Return a ``(num_envs, num_bodies, 3)`` force buffer, mostly zeros.

    Each environment independently receives an impulse with ``probability``,
    applied to one randomly chosen body out of ``body_indices`` (all bodies when
    None), in a uniformly random direction with magnitude up to ``magnitude_n``.
    """
    forces = torch.zeros(
        (int(num_envs), int(num_bodies), 3), dtype=torch.float32, device=device
    )
    probability = float(probability)
    magnitude_n = float(magnitude_n)
    if probability <= 0.0 or magnitude_n <= 0.0:
        return forces

    hit = (
        torch.rand(int(num_envs), device=device, generator=generator)
        < probability
    )
    if not bool(hit.any()):
        return forces
    envs = torch.nonzero(hit, as_tuple=False).squeeze(1)

    if body_indices is None:
        chosen = torch.randint(
            0, int(num_bodies), (envs.numel(),), device=device, generator=generator
        )
    else:
        pick = torch.randint(
            0, len(body_indices), (envs.numel(),), device=device, generator=generator
        )
        chosen = body_indices[pick]

    # Uniform on the sphere, then a uniform magnitude: a fixed magnitude would
    # teach the policy the size of every push it will ever see.
    direction = torch.randn(
        (envs.numel(), 3), device=device, generator=generator
    )
    direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-6)
    scale = torch.rand(
        (envs.numel(), 1), device=device, generator=generator
    ) * magnitude_n
    forces[envs, chosen] = direction * scale
    return forces
