"""
Reveal-IG core (model-agnostic).

For a model f: R^D -> R^C and a target scalar objective (e.g. logit of class k),
computes per-input-dimension attributions by integrating df/dKL * dKL/dt along a
path in (mu, logvar) space from the standard normal prior to a near-deterministic
distribution at the true input value.

The attribution for dimension i is:

    attr_i = integral_0^1 [ df/d_mu_i * mu_final_i
                           + df/d_logvar_i * logvar_final ] dt

Completeness: sum_i attr_i ~ E[f(x_final)] - E[f(x_noise)]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from tqdm import tqdm

from revealig.core.kl import kl_delta
from revealig.core.path import DistributionPath, LinearPath


@dataclass
class AttributionResult:
    """Output of KLIntegratedGradients.attribute()."""

    # Full per-dimension attribution in Reveal-IG sense.
    attr: torch.Tensor

    # Component breakdowns (attr = attr_mu + attr_logvar)
    attr_mu: torch.Tensor
    attr_logvar: torch.Tensor

    # KL divergence of the final distribution from N(0,1), same shape as attr.
    kl_final: torch.Tensor

    # Class index used as the objective.
    target: int

    def completeness_check(self) -> float:
        """
        Returns sum(attr) -- should approximate E[f(final)] - E[f(noise)].
        """
        return float(self.attr.sum().item())


class RevealIG:
    """
    Model-agnostic Reveal-IG.

    Args:
        model:          Any nn.Module. Called as model(x) where x has the same
                        shape as the input passed to attribute(), with an extra
                        batch dimension prepended (n_samples inputs at once).
        n_steps:        Number of integration steps (quadrature points).
        n_samples:      MC samples per step for the gradient expectation.
        sigma_final:    Stddev of the near-deterministic final distribution.
                        Default 1/256 (one 8-bit step).
        path:           Path through (mu, logvar) space. Defaults to LinearPath.
        device:         Torch device. Defaults to the first model parameter's device.
    """

    def __init__(
        self,
        model: nn.Module,
        n_steps: int = 50,
        n_samples: int = 10,
        sigma_final: float = 1.0 / 256.0,
        path: DistributionPath | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.logvar_final = 2.0 * math.log(sigma_final)
        self.path = path or LinearPath()
        self.device = device or next(model.parameters()).device

    def attribute(
        self,
        x: torch.Tensor,
        target: int | Callable[[torch.Tensor], torch.Tensor] | None = None,
        show_progress: bool = False,
    ) -> AttributionResult:
        """
        Compute Reveal-IG attributions for input x.

        Args:
            x:              Input tensor. Shape (D,) or (1, D,...) -- the leading
                            batch dimension is optional and will be squeezed.
            target:         - int: index into model output logits used as objective.
                            - Callable: receives model output (n_samples, *out_shape)
                              and must return a (n_samples,) tensor; the integrator
                              averages over samples. Example for regression:
                              ``target=lambda out: out.squeeze(-1)``
                            - None: uses argmax of model output on x.
            show_progress:  Show a tqdm progress bar over integration steps.

        Returns:
            AttributionResult with attr shape matching x (without batch dim).
        """
        self.model.eval()
        x = x.to(self.device)

        if x.dim() > 1 and x.shape[0] == 1:
            x = x.squeeze(0)

        x_shape = x.shape
        mu_final = self.path.encode(x.detach())
        logvar_final = torch.full_like(mu_final, self.logvar_final)

        objective_fn = self._build_objective(x, target)

        saved_requires_grad = self._disable_model_grad()

        attr_mu_sum = torch.zeros_like(mu_final)
        attr_logvar_sum = torch.zeros_like(mu_final)

        steps = self.path.steps(self.n_steps)
        iterator = tqdm(steps, desc="Reveal-IG", unit="step") if show_progress else steps

        try:
            for t in iterator:
                t_val = float(t)
                g_mu, g_logvar = self._step_gradients(
                    t_val, mu_final, logvar_final, x_shape, objective_fn
                )
                dmu_dt, dlogvar_dt = self.path.derivatives(t_val, mu_final, logvar_final)
                with torch.no_grad():
                    attr_mu_sum.add_(g_mu * dmu_dt)
                    attr_logvar_sum.add_(g_logvar * dlogvar_dt)
        finally:
            self._restore_model_grad(saved_requires_grad)

        with torch.no_grad():
            attr_mu_encoded = attr_mu_sum / self.n_steps
            attr_logvar_encoded = attr_logvar_sum / self.n_steps
            attr_encoded = attr_mu_encoded + attr_logvar_encoded

            attr_mu = self.path.decode_attribution(attr_mu_encoded)
            attr_logvar = self.path.decode_attribution(attr_logvar_encoded)
            attr = self.path.decode_attribution(attr_encoded)

            path_kl = self.path.compute_endpoint_kl(
                x.detach(), torch.full_like(mu_final, self.logvar_final)
            )
            kl = (
                path_kl
                if path_kl is not None
                else kl_delta(x.detach(), torch.full_like(x, self.logvar_final))
            )

        return AttributionResult(
            attr=attr,
            attr_mu=attr_mu,
            attr_logvar=attr_logvar,
            kl_final=kl,
            target=target if isinstance(target, int) else -1,
        )

    def _step_gradients(
        self,
        t: float,
        mu_final: torch.Tensor,
        logvar_final: torch.Tensor,
        x_shape: torch.Size,
        objective_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """At path position t, compute E[df/d_mu] and E[df/d_logvar] via MC sampling."""
        mu_t_val, logvar_t_val = self.path.at(t, mu_final, logvar_final)
        mu_t = mu_t_val.detach().requires_grad_(True)
        logvar_t = logvar_t_val.detach().requires_grad_(True)

        eps = torch.randn(self.n_samples, *x_shape, device=self.device)
        std_t = (0.5 * logvar_t).exp()
        raw_samp = mu_t.unsqueeze(0) + std_t.unsqueeze(0) * eps
        x_samp = self.path.decode_sample(raw_samp)

        obj = objective_fn(x_samp)
        obj.backward()

        return mu_t.grad.clone(), logvar_t.grad.clone()

    def _build_objective(
        self,
        x: torch.Tensor,
        target: int | Callable | None,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return a function (n_samples, *x_shape) -> scalar."""
        if callable(target):
            return lambda x_samp: target(self.model(x_samp)).mean()

        if target is None:
            with torch.no_grad():
                out = self.model(x.unsqueeze(0))
                target = int(out.argmax(dim=-1).item())

        idx = target

        def _obj(x_samp: torch.Tensor) -> torch.Tensor:
            out = self.model(x_samp)
            return out[:, idx].mean()

        return _obj

    def _disable_model_grad(self) -> list[bool]:
        states = [p.requires_grad for p in self.model.parameters()]
        for p in self.model.parameters():
            p.requires_grad_(False)
        return states

    def _restore_model_grad(self, states: list[bool]) -> None:
        for p, s in zip(self.model.parameters(), states):
            p.requires_grad_(s)
