"""Vectorized rollout storage compatible with AnimRL PPO."""

import torch


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        num_obs,
        num_critic_obs,
        num_actions,
        device="cpu",
    ):
        self.device = device
        self.num_obs = num_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        shape = (num_transitions_per_env, num_envs)

        self.observations = torch.zeros(*shape, num_obs, device=device)
        self.privileged_observations = (
            torch.zeros(*shape, num_critic_obs, device=device)
            if num_critic_obs is not None
            else None
        )
        self.actions = torch.zeros(*shape, num_actions, device=device)
        self.dones = torch.zeros(*shape, 1, device=device, dtype=torch.uint8)
        self.rewards = torch.zeros(*shape, 1, device=device)
        self.values = torch.zeros(*shape, 1, device=device)
        self.returns = torch.zeros(*shape, 1, device=device)
        self.advantages = torch.zeros(*shape, 1, device=device)
        self.actions_log_prob = torch.zeros(*shape, 1, device=device)
        self.mu = torch.zeros(*shape, num_actions, device=device)
        self.sigma = torch.zeros(*shape, num_actions, device=device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.step = 0

    def add_transitions(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(
                transition.critic_observations
            )
        self.actions[self.step].copy_(transition.actions)
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(
            transition.actions_log_prob.view(-1, 1)
        )
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = (
                last_values
                if step == self.num_transitions_per_env - 1
                else self.values[step + 1]
            )
            next_is_not_terminal = 1.0 - self.dones[step].float()
            advantage = self.rewards[step] + next_is_not_terminal * gamma * (
                next_values + lam * advantage
            ) - self.values[step]
            self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values
        self.advantages = (
            self.advantages - self.advantages.mean()
        ) / (self.advantages.std() + 1e-8)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        if mini_batch_size <= 0 or batch_size % num_mini_batches != 0:
            raise ValueError(
                "Rollout batch {} must be divisible by {} minibatches".format(
                    batch_size, num_mini_batches
                )
            )
        indices = torch.randperm(batch_size, device=self.device)

        observations = self.observations.flatten(0, 1)
        critic_observations = (
            self.privileged_observations.flatten(0, 1)
            if self.privileged_observations is not None
            else observations
        )
        tensors = (
            observations,
            critic_observations,
            self.actions.flatten(0, 1),
            self.values.flatten(0, 1),
            self.advantages.flatten(0, 1),
            self.returns.flatten(0, 1),
            self.actions_log_prob.flatten(0, 1),
            self.mu.flatten(0, 1),
            self.sigma.flatten(0, 1),
        )
        for _ in range(num_epochs):
            for index in range(num_mini_batches):
                start = index * mini_batch_size
                end = (index + 1) * mini_batch_size
                batch_indices = indices[start:end]
                yield tuple(values[batch_indices] for values in tensors)
