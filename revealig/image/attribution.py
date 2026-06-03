"""
Image-specific wrapper around KLIntegratedGradients.

Handles:
- (1, C, H, W) / (C, H, W) input normalisation
- Per-channel -> per-pixel collapse
- Convenience access to the AttributionResult
"""

from __future__ import annotations

import warnings
from typing import Callable

import torch
import torch.nn as nn

from revealig.core.integrator import AttributionResult, RevealIG
from revealig.core.path import DistributionPath


class ImageAttributor:
    """
    Reveal-IG attribution for image models.

    Args:
        model:        nn.Module expecting input (B, C, H, W).
        n_steps:      Integration steps.
        n_samples:    MC samples per step.
        sigma_final:  Final distribution stddev (default 0.25).
        path:         Path through (mu, logvar) space (default LinearPath).
        device:       Torch device (default: inferred from model).
    """

    def __init__(
        self,
        model: nn.Module,
        n_steps: int = 50,
        n_samples: int = 10,
        sigma_final: float = 0.25,
        path: DistributionPath | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.reveal_ig = RevealIG(
            model=model,
            n_steps=n_steps,
            n_samples=n_samples,
            sigma_final=sigma_final,
            path=path,
            device=device,
        )

    def attribute(
        self,
        x: torch.Tensor,
        target: int | Callable[[torch.Tensor], torch.Tensor] | None = None,
        show_progress: bool = False,
    ) -> "ImageAttributionResult":
        """
        Compute Reveal-IG attributions for an image.

        Args:
            x:              Image tensor, shape (C, H, W) or (1, C, H, W).
            target:         Class index, custom objective callable, or None (argmax).
            show_progress:  Show tqdm progress bar.

        Returns:
            ImageAttributionResult with helper methods for collapsing channels.
        """
        _check_input_stats(x)
        result = self.reveal_ig.attribute(x, target=target, show_progress=show_progress)
        return ImageAttributionResult(result)


def _check_input_stats(x: torch.Tensor) -> None:
    """Warn if the input tensor looks un-normalised."""
    with torch.no_grad():
        flat = x.detach().float()
        if flat.dim() >= 3:
            per_ch = flat.reshape(flat.shape[0], -1)
            mean = per_ch.mean(dim=1).abs().max().item()
            std = per_ch.std(dim=1).mean().item()
        else:
            mean = flat.mean().abs().item()
            std = flat.std().item()

    issues = []
    if mean > 5.0:
        issues.append(f"max channel |mean|={mean:.2f} (expected < 5)")
    if std < 0.1 or std > 5.0:
        issues.append(f"mean channel std={std:.2f} (expected ~1)")

    if issues:
        warnings.warn(
            f"Input may not be normalised to N(0,1): {', '.join(issues)}. "
            "Reveal-IG's prior is N(0,1); consider applying per-channel normalisation "
            "(e.g. ImageNet mean/std) before calling attribute().",
            stacklevel=3,
        )


class ImageAttributionResult:
    """Wraps AttributionResult with image-specific channel-collapse utilities."""

    def __init__(self, result: AttributionResult) -> None:
        self._r = result

    @property
    def attr(self) -> torch.Tensor:
        """(C, H, W) attribution in Reveal-IG sense."""
        return self._r.attr

    @property
    def attr_mu(self) -> torch.Tensor:
        """(C, H, W) mu-component of attribution."""
        return self._r.attr_mu

    @property
    def attr_logvar(self) -> torch.Tensor:
        """(C, H, W) logvar-component of attribution."""
        return self._r.attr_logvar

    @property
    def kl_final(self) -> torch.Tensor:
        """(C, H, W) KL divergence of final distribution from N(0,1)."""
        return self._r.kl_final

    @property
    def target(self) -> int:
        return self._r.target

    @property
    def completeness(self) -> float:
        """Σ attr ≈ E[f(x_final)] − E[f(x_noise)]; ideally close to 1.0."""
        return self._r.completeness_check()

    def attr_map(self, method: str = "sum") -> torch.Tensor:
        """
        Collapse (C, H, W) attribution to (H, W).

        Args:
            method: one of
                - "sum"     : attr.sum(dim=0)       [default, preserves completeness]
                - "absmax"  : abs(attr).max(dim=0)  [sign-preserving channel pick]
                - "sumabs"  : abs(attr).sum(dim=0)
                - "l2"      : attr.pow(2).sum(dim=0).sqrt()

        Returns:
            (H, W) tensor.
        """
        a = self._r.attr
        if method == "absmax":
            abs_a = a.abs()
            idx = abs_a.argmax(dim=0, keepdim=True)
            return a.gather(0, idx).squeeze(0)
        elif method == "sumabs":
            return a.abs().sum(dim=0)
        elif method == "sum":
            return a.sum(dim=0)
        elif method == "l2":
            return a.pow(2).sum(dim=0).sqrt()
        else:
            raise ValueError(f"Unknown collapse method '{method}'")

    def attr_map_clipped(
        self,
        method: str = "sum",
        clip_percentile: float = 99.0,
    ) -> torch.Tensor:
        """Collapse and clip at a percentile, then normalise to [0, 1]."""
        m = self.attr_map(method)
        clip_val = torch.quantile(m.abs(), clip_percentile / 100.0)
        m = m.clamp(-clip_val, clip_val)
        lo, hi = m.min(), m.max()
        if hi > lo:
            m = (m - lo) / (hi - lo)
        return m

    def __repr__(self) -> str:
        shape = tuple(self._r.attr.shape)
        return (
            f"ImageAttributionResult(shape={shape}, target={self.target}, "
            f"completeness={self._r.completeness_check():.4f})"
        )
