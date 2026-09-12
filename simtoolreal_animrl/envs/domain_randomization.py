"""Training-time disturbances, so the policy is not tuned to one exact robot.

The actuator, latency, observation, and reset vocabulary comes from
``scripts/probe_robustness.py``. Physics-property variation and short external
cube wrenches extend that measured baseline.

Everything is drawn together rather than one axis per episode, so the policy
learns combinations of errors rather than isolated perturbations.

Kept free of isaacgym so the draws can be tested without a simulator.
"""

import math

import torch


# sigma_dq = sqrt(2) * control_hz * sigma_q is the cost of differentiating an
# encoder once per control step, so the velocity noise is not a free parameter:
# it follows from the position noise and the rate.
def velocity_noise_for_position_noise(position_noise_rad, control_hz):
    """The dq noise implied by differentiating a noisy q at ``control_hz``."""
    return math.sqrt(2.0) * float(control_hz) * float(position_noise_rad)


def log_uniform_scales(shape, log2_halfwidth, device, generator=None):
    """Multiplicative factors uniform in log2 over ``+- log2_halfwidth``.

    Drawn in log space so that halving and doubling a gain are equally likely;
    a uniform draw on the linear scale would favour the stiff side.
    """
    halfwidth = float(log2_halfwidth)
    if halfwidth <= 0.0:
        return torch.ones(shape, device=device)
    exponent = (
        torch.rand(shape, device=device, generator=generator) * 2.0 - 1.0
    ) * halfwidth
    return torch.pow(2.0, exponent)


def random_vectors_in_ball(count, max_magnitude, device, generator=None):
    """Sample isotropic 3D vectors with magnitudes uniform on ``[0, max]``."""
    count = int(count)
    maximum = float(max_magnitude)
    if count < 0:
        raise ValueError("count cannot be negative")
    if not math.isfinite(maximum) or maximum < 0.0:
        raise ValueError("max_magnitude must be finite and non-negative")
    if count == 0 or maximum == 0.0:
        return torch.zeros((count, 3), device=device)
    directions = torch.randn((count, 3), device=device, generator=generator)
    directions /= torch.linalg.vector_norm(
        directions, dim=1, keepdim=True
    ).clamp_min(1e-12)
    magnitudes = torch.rand(
        (count, 1), device=device, generator=generator
    ) * maximum
    return directions * magnitudes


def _validated_range(cfg, name, *, strictly_positive=False, upper_limit=None):
    values = getattr(cfg, name)
    if len(values) != 2:
        raise ValueError("domain_randomization.{} must contain two values".format(name))
    lower, upper = (float(values[0]), float(values[1]))
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError(
            "domain_randomization.{} must be a finite ordered range".format(name)
        )
    if (strictly_positive and lower <= 0.0) or (not strictly_positive and lower < 0.0):
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ValueError(
            "domain_randomization.{} must be {}".format(name, qualifier)
        )
    if upper_limit is not None and upper > upper_limit:
        raise ValueError(
            "domain_randomization.{} cannot exceed {}".format(name, upper_limit)
        )
    return (lower, upper)


def resolve_settings(cfg):
    """Validate the configuration block and return it as a plain dict."""
    if not bool(getattr(cfg, "enabled", False)):
        return None
    fields = (
        "pd_log2_halfwidth",
        "obs_q_noise_rad",
        "obs_q_bias_rad",
        "obs_dq_noise_rad_s",
        "obs_target_noise_rad",
        "init_q_offset_rad",
        "init_dq_offset_rad_s",
        "gravity_xy_max_m_s2",
        "external_force_max_n",
        "external_torque_max_nm",
    )
    settings = {}
    for name in fields:
        value = float(getattr(cfg, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "domain_randomization.{} must be finite and non-negative".format(
                    name
                )
            )
        settings[name] = value
    delay = int(getattr(cfg, "action_delay_max_steps"))
    if delay < 0:
        raise ValueError(
            "domain_randomization.action_delay_max_steps cannot be negative"
        )
    settings["action_delay_max_steps"] = delay
    range_fields = (
        ("object_mass_scale_range", True, None),
        ("object_inertia_scale_range", True, None),
        ("object_friction_range", False, None),
        ("object_restitution_range", False, 1.0),
        ("robot_friction_scale_range", True, None),
        ("table_friction_range", False, None),
        ("gravity_z_scale_range", True, None),
    )
    for name, strictly_positive, upper_limit in range_fields:
        settings[name] = _validated_range(
            cfg,
            name,
            strictly_positive=strictly_positive,
            upper_limit=upper_limit,
        )
    probability = float(getattr(cfg, "external_wrench_probability_per_step"))
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(
            "domain_randomization.external_wrench_probability_per_step "
            "must be in [0, 1]"
        )
    settings["external_wrench_probability_per_step"] = probability
    duration = int(getattr(cfg, "external_wrench_duration_steps"))
    if duration <= 0:
        raise ValueError(
            "domain_randomization.external_wrench_duration_steps must be positive"
        )
    settings["external_wrench_duration_steps"] = duration
    if bool(getattr(cfg, "couple_velocity_noise_to_position_noise", True)):
        settings["obs_dq_noise_rad_s"] = velocity_noise_for_position_noise(
            settings["obs_q_noise_rad"], float(getattr(cfg, "control_hz", 60.0))
        )
    return settings
