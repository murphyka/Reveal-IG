"""
Pool-based Reveal-IG for tabular data.

Each feature's soft assignment over the training pool is parameterized by a
temperature τ_i. The integration path moves from high entropy (near-uniform
assignment, s ≈ s_start) to low entropy (concentrated at the test point,
s ≈ s_end) in entropy-fraction space.

Attribution for feature i:
    attr_i = ∫_{s_start}^{s_end} E[∂f/∂log τ_i] · (d log τ_i / ds) ds

The integral is approximated with a uniform grid of n_steps quadrature
points (Riemann midpoints), each estimated with n_samples MC draws.

Closed-form gradient (no autograd required in the inner loop):
    ∂p_k/∂log τ  =  p_k · (d_k − ⟨d⟩_p) / τ
    ∂H/∂log τ    = −Σ_v (∂P_v/∂log τ) · log P_v
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from revealig.tabular.pool import IdentityPoolCache


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class TabularAttributionResult:
    """Output of TabularAttributor.attribute()."""

    attr: torch.Tensor  # (D,) feature attributions
    pool_size: int      # number of pool members (including the test point)
    n_steps: int        # number of integration steps used


# ── Internal helpers ──────────────────────────────────────────────────────────

def _assignment_probs(sq_dists: torch.Tensor, log_tau: torch.Tensor) -> torch.Tensor:
    """
    Softmax assignment of the test point over M pool members.

    p_k = softmax(-d_k / τ)_k,  where d_k is the squared distance from
    the test point to pool member k in feature space.
    """
    return F.softmax(-sq_dists / log_tau.exp(), dim=0)


def _value_entropy(
    probs: torch.Tensor,
    value_inverse: torch.Tensor,
    n_unique_values: int,
) -> torch.Tensor:
    """
    Entropy of the value-collapsed assignment distribution.

    Pool members sharing the same feature value are merged: their assignment
    probabilities are summed before computing entropy. This gives the entropy
    of the implicit distribution over distinct observed values.
    """
    value_probs = torch.zeros(n_unique_values, device=probs.device, dtype=probs.dtype)
    value_probs.scatter_add_(0, value_inverse, probs)
    safe_probs = value_probs.clamp_min(1e-30)
    return -(safe_probs * safe_probs.log()).sum()


def _find_log_tau(
    sq_dists: torch.Tensor,
    value_inverse: torch.Tensor,
    n_unique_values: int,
    target_entropy: float,
    lo: float = -40.0,
    hi: float = 40.0,
    n_iter: int = 80,
) -> float:
    """
    Binary-search for log τ such that the value-collapsed entropy equals
    target_entropy.

    Entropy is monotone increasing in log τ:
      log τ → −∞ : assignment concentrates on the test value → H = 0
      log τ → +∞ : assignment is uniform over the pool    → H = H_max

    Runs on CPU regardless of sq_dists's device: the bisection is a scalar
    root-find and each .item() call on a GPU tensor forces a device sync,
    making the loop ~30× slower than necessary.
    """
    sq_dists_cpu    = sq_dists.cpu()
    value_inv_cpu   = value_inverse.cpu()
    with torch.no_grad():
        for _ in range(n_iter):
            mid     = 0.5 * (lo + hi)
            log_tau = torch.tensor(mid)
            entropy = _value_entropy(
                _assignment_probs(sq_dists_cpu, log_tau),
                value_inv_cpu,
                n_unique_values,
            ).item()
            if entropy < target_entropy:
                lo = mid
            else:
                hi = mid
    return 0.5 * (lo + hi)


@dataclass
class _StepState:
    """Pre-computed quantities at one quadrature point on the entropy path."""
    assignment_probs: list[torch.Tensor]  # per-feature softmax distributions, each (M,)
    assignment_grad:  torch.Tensor        # (D, M)  d(assignment_k)/d(log τ_i) per feature
    d_log_tau_d_s:    np.ndarray          # (D,)    chain-rule factor ds → d log τ


def _compute_step_state(
    cache: IdentityPoolCache,
    log_temperatures: np.ndarray,
    H_max: torch.Tensor,
    device: torch.device,
) -> _StepState:
    """
    Compute the gradient weights and chain-rule factor for one path position.

    For each feature i at the given log τ_i:
      - assignment p_k  =  softmax(-d_k / τ_i)
      - ∂p_k/∂log τ_i  =  p_k · (d_k − ⟨d⟩_p) / τ_i
      - ∂H_i/∂log τ_i  =  −Σ_v (∂P_v/∂log τ_i) · log P_v
      - d log τ_i / ds  =  H_max_i / (∂H_i/∂log τ_i)
    """
    D = cache.D
    assignment_probs_list = []
    assignment_grad_list  = []
    dH_d_log_tau          = torch.zeros(D, device=device)

    for i in range(D):
        sq_dists    = cache.test_sq_dists[i]
        temperature = float(np.exp(log_temperatures[i]))

        assignment  = F.softmax(-sq_dists / temperature, dim=0)
        mean_sq_dist = (assignment * sq_dists).sum()
        grad_assignment = assignment * (sq_dists - mean_sq_dist) / temperature

        # Collapse to unique values to compute ∂H/∂log τ
        value_probs = torch.zeros(cache.K[i], device=device)
        value_probs.scatter_add_(0, cache.inverse[i], assignment)
        value_prob_grad = torch.zeros(cache.K[i], device=device)
        value_prob_grad.scatter_add_(0, cache.inverse[i], grad_assignment)

        dH_d_log_tau[i] = -(value_prob_grad * value_probs.clamp_min(1e-30).log()).sum()

        assignment_probs_list.append(assignment)
        assignment_grad_list.append(grad_assignment)

    dH_d_log_tau = dH_d_log_tau.clamp(min=1e-9)
    d_log_tau_d_s = (H_max / dH_d_log_tau).detach().cpu().numpy()

    return _StepState(
        assignment_probs = [p.detach() for p in assignment_probs_list],
        assignment_grad  = torch.stack(assignment_grad_list).detach(),  # (D, M)
        d_log_tau_d_s    = d_log_tau_d_s,
    )


def _mc_gradient_estimate(
    model: nn.Module,
    X_pool: torch.Tensor,
    cache: IdentityPoolCache,
    n_samples: int,
    assignment_probs: list[torch.Tensor],
    assignment_grad: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    MC estimate of E[∂f/∂log τ_i] for all features simultaneously.

    Each sample draws one pool index per feature (j_i ~ p_i), constructs a
    baseline input x_base = (X_pool[j_i, i])_i, then evaluates the model on
    a batch of D·M inputs where each block of M rows varies one feature across
    all pool values while holding the others fixed at x_base. This covers all
    D gradient components in a single forward pass.

    Returns the sum of gradient estimates over n_samples draws, shape (D,).
    The caller divides by n_samples to get the mean.
    """
    M, D = cache.M, cache.D
    grad_sum      = torch.zeros(D, device=device)
    feature_range = torch.arange(D, device=device)

    for _ in range(n_samples):
        # Draw one pool index per feature from its assignment distribution
        base_indices = torch.stack([
            torch.multinomial(assignment_probs[i], 1)[0] for i in range(D)
        ])
        x_base = X_pool[base_indices, feature_range]       # (D,)

        # Build evaluation batch: repeat x_base D*M times, then
        # for block i replace feature i with all M pool values
        x_batch = x_base.unsqueeze(0).expand(D * M, D).contiguous()
        for i in range(D):
            x_batch[i * M:(i + 1) * M, i] = X_pool[:, i]

        with torch.no_grad():
            f_vals = model(x_batch).reshape(D, M)          # (D, M)

        grad_estimate = (assignment_grad * f_vals).sum(dim=1)  # (D,)
        grad_sum += grad_estimate

    return grad_sum


# ── Public API ────────────────────────────────────────────────────────────────

class TabularAttributor:
    """
    Pool-based Reveal-IG for tabular data.

    Computes feature attributions by integrating along an entropy path from a
    near-uniform reference distribution (s = s_start) to a distribution
    concentrated at the test point (s = s_end). At each path position s,
    feature i's soft assignment over the pool has temperature τ_i(s) chosen
    so that the value-collapsed entropy equals s · H_max_i.

    The integral uses a uniform grid of n_steps Riemann midpoints, each
    estimated with n_samples MC draws. The same total-budget MC allocation
    is applied at every step.

    Args:
        model:     nn.Module called as model(x) with x of shape (batch, D).
                   For regression, output shape (batch, 1) or (batch,).
        n_steps:   Number of quadrature points along the entropy path.
        n_samples: Number of MC samples per step.
        s_start:   Starting entropy fraction (near 1 = near-uniform over pool).
        s_end:     Ending entropy fraction (near 0 = concentrated at test point).
        device:    Torch device. Defaults to the device of the first model parameter.

    Example::

        from revealig.tabular import TabularAttributor

        attributor = TabularAttributor(model, n_steps=40, n_samples=40)
        result = attributor.attribute(x_test, X_train_subset)
        # result.attr: (D,) feature attributions
        # sum(result.attr) ≈ E[f | path_end] − E[f | path_start]
    """

    def __init__(
        self,
        model: nn.Module,
        n_steps: int = 40,
        n_samples: int = 40,
        s_start: float = 0.99,
        s_end: float = 0.05,
        device: torch.device | None = None,
    ) -> None:
        self.model    = model
        self.n_steps  = n_steps
        self.n_samples = n_samples
        self.s_start  = s_start
        self.s_end    = s_end
        self.device   = device or next(model.parameters()).device

    def attribute(
        self,
        x_test: torch.Tensor,
        X_background: torch.Tensor,
        show_progress: bool = False,
    ) -> TabularAttributionResult:
        """
        Compute Reveal-IG attributions for x_test relative to X_background.

        X_background should be a representative sample from the training
        distribution (e.g. a random subset of training data). x_test is
        prepended to form the pool, so it is always a pool member at index 0.

        sum(result.attr) ≈ E[f | path_end] − E[f | path_start]

        Args:
            x_test:        (D,) tensor — the input to explain.
            X_background:  (M, D) tensor — reference pool (training subset).
            show_progress: If True, show a tqdm progress bar over steps.

        Returns:
            TabularAttributionResult with .attr of shape (D,).
        """
        dev = self.device
        x_test       = x_test.to(dev)
        X_background = X_background.to(dev)

        X_pool = torch.cat([x_test.unsqueeze(0), X_background], dim=0)
        cache  = IdentityPoolCache(X_pool, test_idx_in_pool=0)

        attr_np = self._integrate(cache, X_pool, show_progress)

        return TabularAttributionResult(
            attr      = torch.from_numpy(attr_np).to(dev),
            pool_size = cache.M,
            n_steps   = self.n_steps,
        )

    def _integrate(
        self,
        cache: IdentityPoolCache,
        X_pool: torch.Tensor,
        show_progress: bool,
    ) -> np.ndarray:
        dev   = self.device
        D     = cache.D
        H_max = torch.tensor(cache.H_max, device=dev)

        # Uniform quadrature grid: midpoints and signed step sizes
        edges         = np.linspace(self.s_start, self.s_end, self.n_steps + 1)
        step_midpoints = 0.5 * (edges[:-1] + edges[1:])
        step_sizes     = np.diff(edges)   # negative, since s_start > s_end

        attr = np.zeros(D)
        steps = tqdm(range(self.n_steps), desc="Reveal-IG") if show_progress \
                else range(self.n_steps)

        for k in steps:
            # Find temperature for each feature at this entropy level
            log_temperatures = np.array([
                _find_log_tau(
                    cache.test_sq_dists[i],
                    cache.inverse[i],
                    cache.K[i],
                    target_entropy=step_midpoints[k] * cache.H_max[i],
                )
                for i in range(D)
            ])

            step = _compute_step_state(cache, log_temperatures, H_max, dev)

            grad_sum = _mc_gradient_estimate(
                self.model, X_pool, cache, self.n_samples,
                step.assignment_probs, step.assignment_grad, dev,
            )
            mean_grad = (grad_sum / self.n_samples).detach().cpu().numpy()

            # attr_i += E[∂f/∂log τ_i] · (d log τ_i / ds) · ds
            attr += mean_grad * step.d_log_tau_d_s * step_sizes[k]

        return attr
