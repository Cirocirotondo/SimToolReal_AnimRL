"""Probability distributions retained from AnimRL's policy implementation."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import nn
from torch.distributions import Normal


class Distribution(ABC):
    def __init__(self):
        super().__init__()
        self.distribution = None

    @abstractmethod
    def proba_distribution_net(self, *args, **kwargs):
        pass

    @abstractmethod
    def proba_distribution(self, *args, **kwargs):
        pass

    @abstractmethod
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def entropy(self) -> Optional[torch.Tensor]:
        pass

    @abstractmethod
    def sample(self) -> torch.Tensor:
        pass

    @abstractmethod
    def mode(self) -> torch.Tensor:
        pass

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        return self.mode() if deterministic else self.sample()

    @abstractmethod
    def actions_from_params(self, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def log_prob_from_params(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    if len(tensor.shape) > 1:
        return tensor.sum(dim=1)
    return tensor.sum()


class DiagGaussianDistribution(Distribution):
    """Unsquashed diagonal Gaussian used by the original AnimRL PPO."""

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0):
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        log_std = nn.Parameter(
            torch.ones(self.action_dim) * log_std_init, requires_grad=True
        )
        return mean_actions, log_std

    def proba_distribution(
        self, mean_actions: torch.Tensor, log_std: torch.Tensor
    ):
        action_std = torch.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return sum_independent_dims(self.distribution.log_prob(actions))

    def entropy(self) -> torch.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> torch.Tensor:
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        return self.distribution.mean

    def actions_from_params(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self, mean_actions: torch.Tensor, log_std: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(mean_actions, log_std)
        return actions, self.log_prob(actions)
