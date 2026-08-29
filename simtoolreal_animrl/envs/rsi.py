"""Reference-state initialization sampling shared by training and evaluation."""

from typing import Optional, Tuple

import torch


def resolve_rsi_settings(env_cfg, reference_last_index: int) -> Tuple[str, int, int, float]:
    """Validate and return the configured inclusive RSI sampling ranges."""
    distribution = str(env_cfg.reference_init_distribution)
    if distribution not in ("uniform", "pregrasp_mixture"):
        raise ValueError(
            "Unsupported reference_init_distribution {!r}".format(distribution)
        )

    available_max = int(reference_last_index) - 1
    max_start = int(env_cfg.rsi_max_start_index)
    pregrasp_start = int(env_cfg.rsi_pregrasp_start_index)
    early_probability = float(env_cfg.rsi_early_probability)
    if not 0 <= max_start <= available_max:
        raise ValueError(
            "env.rsi_max_start_index must lie in [0, {}]".format(available_max)
        )
    if not 0 <= pregrasp_start <= max_start:
        raise ValueError(
            "env.rsi_pregrasp_start_index must lie in [0, {}]".format(max_start)
        )
    if not 0.0 <= early_probability <= 1.0:
        raise ValueError("env.rsi_early_probability must lie in [0, 1]")
    if distribution == "pregrasp_mixture" and early_probability > 0.0:
        if pregrasp_start == 0:
            raise ValueError(
                "A positive early RSI probability requires a non-empty early range"
            )
    return distribution, max_start, pregrasp_start, early_probability


def sample_rsi_indices(
    count: int,
    device: torch.device,
    distribution: str,
    max_start: int,
    pregrasp_start: int,
    early_probability: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample RSI frames, with the mixture probability interpreted per reset."""
    count = int(count)
    if count < 0:
        raise ValueError("RSI sample count cannot be negative")
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if distribution == "uniform":
        return torch.randint(
            0,
            int(max_start) + 1,
            (count,),
            dtype=torch.long,
            device=device,
            generator=generator,
        )
    if distribution != "pregrasp_mixture":
        raise ValueError("Unsupported RSI distribution {!r}".format(distribution))

    # Draw the pre-grasp cohort first, then replace the configured fraction by
    # early-motion starts. This avoids ever sampling the unstable post-830
    # states while keeping every frame in both configured ranges reachable.
    indices = torch.randint(
        int(pregrasp_start),
        int(max_start) + 1,
        (count,),
        dtype=torch.long,
        device=device,
        generator=generator,
    )
    if early_probability <= 0.0:
        return indices
    early = torch.rand(count, device=device, generator=generator) < float(
        early_probability
    )
    early_indices = torch.randint(
        0,
        int(pregrasp_start),
        (count,),
        dtype=torch.long,
        device=device,
        generator=generator,
    )
    return torch.where(early, early_indices, indices)
