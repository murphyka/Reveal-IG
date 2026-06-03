"""
Path parameterization in (mu, logvar) space.

A path defines how the distribution parameters evolve from the prior
(mu=0, logvar=0, i.e. N(0,1)) to the final distribution at t=1.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import torch


class DistributionPath(ABC):
    """Abstract base for paths through (mu, logvar) space."""

    @abstractmethod
    def at(
        self,
        t: float,
        mu_final: torch.Tensor,
        logvar_final: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu_t, logvar_t) at position t in [0, 1]."""
        ...

    @abstractmethod
    def steps(self, n: int) -> torch.Tensor:
        """Return the t values to evaluate for numerical integration (length n)."""
        ...

    def derivatives(
        self,
        t: float,
        mu_final: torch.Tensor,
        logvar_final: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (d_mu_dt, d_logvar_dt) at time t."""
        raise NotImplementedError

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Transform model-input x into the space where (mu, logvar) live."""
        return x

    def decode_sample(self, raw_sample: torch.Tensor) -> torch.Tensor:
        """Transform a batch of reparameterized samples back to model input space."""
        return raw_sample

    def compute_endpoint_kl(
        self,
        x: torch.Tensor,
        logvar_final: torch.Tensor,
    ) -> torch.Tensor | None:
        """Per-element KL divergence of the endpoint distribution from the reference."""
        return None

    def decode_attribution(self, attr_encoded: torch.Tensor) -> torch.Tensor:
        """Transform attribution from encoded space back to reporting space."""
        return attr_encoded


class LinearPath(DistributionPath):
    """
    Linear interpolation in (mu, logvar) space.

    mu(t)     = t * mu_final
    logvar(t) = t * logvar_final
    """

    def at(
        self,
        t: float,
        mu_final: torch.Tensor,
        logvar_final: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return t * mu_final, t * logvar_final

    def derivatives(
        self,
        t: float,
        mu_final: torch.Tensor,
        logvar_final: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return mu_final, logvar_final

    def steps(self, n: int) -> torch.Tensor:
        return torch.linspace(0.5 / n, 1.0 - 0.5 / n, n)
