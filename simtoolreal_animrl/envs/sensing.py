"""Sensor realism: noisy measurements and a delayed command path.

Two things every real controller has and a simulator does not, taken from a
config that produced a notably smooth policy on another machine:

Observation noise. The policy reads exact joint angles and velocities here,
while hardware reads encoder counts and a differentiated velocity. This matters
beyond robustness: a high-gain reactive policy amplifies measurement noise into
action noise and is punished for it, so noise suppresses chatter for a reason,
where an action-rate penalty only forbids it. Velocity gets the larger share
because differentiating a quantised position is where real noise lives.

Action delay. A command issued now reaches a real joint one or more control
steps later, so an aggressive corrector overshoots. Training with delay forces
the low-gain behaviour that survives the transfer.

Kept free of isaacgym so both are testable without a simulation.
"""

import torch


def add_observation_noise(
    positions,
    velocities,
    position_noise_rad,
    velocity_noise_rad_s,
    position_bias_rad=0.0,
    bias=None,
    generator=None,
):
    """Return noisy ``(positions, velocities)``.

    ``bias`` is a fixed per-environment, per-joint offset standing in for an
    encoder's zero error: it is constant for an episode rather than resampled,
    because a bias the policy could average away over a few steps is not a bias.
    """
    positions = positions.clone()
    velocities = velocities.clone()
    if float(position_noise_rad) > 0.0:
        positions += torch.randn(
            positions.shape, device=positions.device, dtype=positions.dtype,
            generator=generator,
        ) * float(position_noise_rad)
    if bias is not None and float(position_bias_rad) > 0.0:
        positions += bias
    if float(velocity_noise_rad_s) > 0.0:
        velocities += torch.randn(
            velocities.shape, device=velocities.device, dtype=velocities.dtype,
            generator=generator,
        ) * float(velocity_noise_rad_s)
    return positions, velocities


def sample_position_bias(
    num_envs, num_joints, position_bias_rad, device, generator=None
):
    """One constant offset per environment per joint, uniform in +-bias."""
    if float(position_bias_rad) <= 0.0:
        return None
    noise = torch.rand(
        (int(num_envs), int(num_joints)), device=device, generator=generator
    )
    return (noise * 2.0 - 1.0) * float(position_bias_rad)


class ActionDelay:
    """A per-environment delay of 0..max_steps control steps.

    The delay is drawn once per environment rather than per step: a delay that
    changes every step is jitter, which averages out, while a real control path
    has a latency that is fixed and unknown.
    """

    def __init__(self, num_envs, num_actions, max_steps, device, generator=None):
        self.max_steps = int(max_steps)
        if self.max_steps < 0:
            raise ValueError("Action delay cannot be negative")
        self.num_envs = int(num_envs)
        self.device = device
        if self.max_steps == 0:
            self.buffer = None
            self.steps = None
            return
        # buffer[k] holds the action from k steps ago.
        self.buffer = torch.zeros(
            (self.max_steps + 1, self.num_envs, int(num_actions)),
            dtype=torch.float32,
            device=device,
        )
        self.steps = torch.randint(
            0, self.max_steps + 1, (self.num_envs,), device=device,
            generator=generator,
        )

    def __call__(self, actions):
        """Push this step's actions in, return what each environment receives."""
        if self.buffer is None:
            return actions
        self.buffer = torch.roll(self.buffer, shifts=1, dims=0)
        self.buffer[0] = actions
        rows = torch.arange(self.num_envs, device=self.device)
        return self.buffer[self.steps, rows]

    def reset(self, env_ids, actions=None):
        """Clear an environment's history so a reset does not leak old commands."""
        if self.buffer is None:
            return
        if actions is None:
            self.buffer[:, env_ids] = 0.0
        else:
            self.buffer[:, env_ids] = actions.unsqueeze(0)
