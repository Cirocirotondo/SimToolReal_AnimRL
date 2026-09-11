"""Per-environment physical variation, for sim2sim and sim2real transfer.

Written after a measured failure: pg830_blind512_n256, the *roughest* policy
trained here (rms action rate 2.53), transferred to MuJoCo well, while
blind_quiet2 and adapt_sigma -- 25x and 47x smoother by the Isaac Gym metrics --
transferred badly with the same PD gains in both simulators. The smooth policies
grip at 1.6-2.9 N where the rough one uses 6.3 N, and they command 30-60% more
drive deflection to do it. Fine distinctions like those are properties of one
contact model, not of the task.

So the smoothness was partly bought by fitting the simulator. Randomising the
parameters the policy cannot observe removes that option: a policy cannot tune
itself to a friction coefficient that changes every environment.

Sampling is per environment at creation, not per episode: Isaac Gym applies DOF
and shape properties when an actor is built, and with hundreds of environments
the population covers the range on every iteration anyway.
"""

import numpy as np


class DomainRandomization:
    """Samples one physical configuration per environment.

    Every range is multiplicative around 1.0 and a range of 0.0 disables that
    parameter, so an unrandomised run reproduces the fixed values exactly.
    """

    def __init__(self, cfg, num_envs, seed=0):
        self.enabled = bool(getattr(cfg, "enabled", False))
        self.num_envs = int(num_envs)
        self._rng = np.random.default_rng(int(seed))
        self.ranges = {
            "arm_stiffness": float(getattr(cfg, "arm_stiffness_range", 0.0)),
            "arm_damping": float(getattr(cfg, "arm_damping_range", 0.0)),
            "hand_stiffness": float(getattr(cfg, "hand_stiffness_range", 0.0)),
            "hand_damping": float(getattr(cfg, "hand_damping_range", 0.0)),
            "fingertip_friction": float(
                getattr(cfg, "fingertip_friction_range", 0.0)
            ),
            "object_friction": float(getattr(cfg, "object_friction_range", 0.0)),
            "object_mass": float(getattr(cfg, "object_mass_range", 0.0)),
            "table_friction": float(getattr(cfg, "table_friction_range", 0.0)),
            "robot_link_mass": float(getattr(cfg, "robot_link_mass_range", 0.0)),
        }
        self.robot_impulse_probability = float(
            getattr(cfg, "robot_impulse_probability", 0.0)
        )
        self.robot_impulse_n = float(getattr(cfg, "robot_impulse_n", 0.0))
        self.object_impulse_probability = float(
            getattr(cfg, "object_impulse_probability", 0.0)
        )
        self.object_impulse_n = float(getattr(cfg, "object_impulse_n", 0.0))
        self.critic_observes_parameters = bool(
            getattr(cfg, "critic_observes_parameters", False)
        )
        self.obs_q_noise_rad = float(getattr(cfg, "obs_q_noise_rad", 0.0))
        self.obs_dq_noise_rad_s = float(getattr(cfg, "obs_dq_noise_rad_s", 0.0))
        self.obs_q_bias_rad = float(getattr(cfg, "obs_q_bias_rad", 0.0))
        self.action_delay_max_steps = int(
            getattr(cfg, "action_delay_max_steps", 0)
        )
        for name, value in self.ranges.items():
            if value < 0.0 or value >= 1.0:
                raise ValueError(
                    "Randomisation range for {} must lie in [0, 1)".format(name)
                )
        self.samples = self._draw()

    def _draw(self):
        """One multiplier per environment per parameter."""
        samples = {}
        for name, spread in self.ranges.items():
            if not self.enabled or spread <= 0.0:
                samples[name] = np.ones(self.num_envs, dtype=np.float64)
            else:
                samples[name] = self._rng.uniform(
                    1.0 - spread, 1.0 + spread, size=self.num_envs
                )
        return samples

    def multiplier(self, name, env_index):
        return float(self.samples[name][int(env_index)])

    def privileged_row(self, env_index):
        """The sampled multipliers, centred on zero, for the critic.

        The critic is discarded at deployment, so telling it which environment
        it is in costs the shipped policy nothing -- and without it the value
        function sees identical observations from environments with different
        friction and has to average over outcomes it cannot explain. That
        unexplained variance lands in the advantages the actor learns from.
        """
        return np.array(
            [self.samples[name][int(env_index)] - 1.0 for name in sorted(self.ranges)],
            dtype=np.float32,
        )

    @property
    def privileged_dim(self):
        if not (self.enabled and self.critic_observes_parameters):
            return 0
        return len(self.ranges)

    def privileged_table(self, device=None):
        """All environments' multipliers as one (num_envs, dim) tensor."""
        import torch

        import numpy as np

        if self.privileged_dim == 0:
            return None
        rows = np.stack(
            [self.samples[name] - 1.0 for name in sorted(self.ranges)], axis=1
        ).astype(np.float32)
        return torch.as_tensor(rows, device=device)

    @property
    def observation_noise_enabled(self):
        return self.enabled and (
            self.obs_q_noise_rad > 0.0
            or self.obs_dq_noise_rad_s > 0.0
            or self.obs_q_bias_rad > 0.0
        )

    @property
    def action_delay_steps(self):
        return self.action_delay_max_steps if self.enabled else 0

    @property
    def impulses_enabled(self):
        return self.enabled and (
            (self.robot_impulse_probability > 0.0 and self.robot_impulse_n > 0.0)
            or (
                self.object_impulse_probability > 0.0
                and self.object_impulse_n > 0.0
            )
        )

    def summary(self):
        return {
            name: (float(values.min()), float(values.max()))
            for name, values in sorted(self.samples.items())
        }
