"""
KL divergence utilities for Gaussian distributions against N(0,1).

KL(N(mu, exp(logvar)) || N(0,1)) = 0.5 * (exp(logvar) + mu^2 - 1 - logvar)

This is computed per-element; summing over all pixel-channels gives total KL.
"""

import torch


def gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Per-element KL divergence from N(mu, exp(logvar)) to N(0, 1).

    Args:
        mu:     any shape
        logvar: same shape as mu

    Returns:
        KL tensor of same shape, all values >= 0.
    """
    return 0.5 * (logvar.exp() + mu.pow(2) - 1.0 - logvar)


def kl_delta(mu_final: torch.Tensor, logvar_final: torch.Tensor) -> torch.Tensor:
    """
    Total KL change along the path: KL at t=1 minus KL at t=0.
    KL at t=0 is 0 because the prior equals N(0,1).
    """
    return gaussian_kl(mu_final, logvar_final)
