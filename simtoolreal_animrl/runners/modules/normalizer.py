"""Empirical observation normalizer retained from AnimRL."""

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    def __init__(self, shape, eps=1e-2, until=None):
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.count = 0

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, values):
        if self.training:
            self.update(values)
        return (values - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, values):
        if self.until is not None and self.count >= self.until:
            return
        count_x = values.shape[0]
        self.count += count_x
        rate = count_x / self.count
        var_x = torch.var(values, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(values, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (
            var_x - self._var + delta_mean * (mean_x - self._mean)
        )
        self._std = torch.sqrt(self._var)
