"""
Adaptive stopping criterion for Reveal-IG.

find_sigma_stop() binary-searches for the largest sigma such that the model's
expected logit for the target class under Gaussian noise N(0, sigma^2 I) stays
above tau times the clean logit. This is the most noise the image can absorb
while the model is still confident about the correct class.

Using sigma_stop as sigma_final in ImageAttributor makes the path end at a
"perceptually noisy but still correctly classified" distribution rather than
the arbitrarily tight 1/256.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def find_sigma_stop(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    tau: float = 0.95,
    n_samples: int = 64,
    n_iter: int = 15,
    sigma_lo: float = 0.0,
    sigma_hi: float = 2.0,
) -> float:
    """
    Find the largest sigma such that E[f(x + N(0,sigma^2 I))_target] >= tau * f(x_clean)_target.

    Binary search over sigma in [sigma_lo, sigma_hi]. Each iteration evaluates
    n_samples noisy copies of x in a single forward pass.

    Args:
        model:      nn.Module, called as model(x_batch) -> (B, n_classes) logits.
        x:          Input image, shape (C, H, W) or (1, C, H, W).
        target:     Class index to evaluate.
        tau:        Confidence retention threshold (default 0.95).
        n_samples:  MC samples per binary-search step.
        n_iter:     Number of bisection steps.
        sigma_lo:   Lower bound of the search range.
        sigma_hi:   Upper bound of the search range.

    Returns:
        sigma_stop: the largest sigma satisfying the confidence threshold.
    """
    model.eval()

    if x.dim() == 4 and x.shape[0] == 1:
        x = x.squeeze(0)
    device = next(model.parameters()).device
    x = x.to(device)

    with torch.no_grad():
        f_clean = model(x.unsqueeze(0))[0, target].item()

    threshold = tau * f_clean

    lo, hi = sigma_lo, sigma_hi

    with torch.no_grad():
        for _ in range(n_iter):
            mid = (lo + hi) / 2.0
            eps = torch.randn(n_samples, *x.shape, device=device)
            x_noisy = x.unsqueeze(0) + mid * eps
            f_noisy = model(x_noisy)[:, target].mean().item()
            if f_noisy >= threshold:
                lo = mid
            else:
                hi = mid

    return lo
