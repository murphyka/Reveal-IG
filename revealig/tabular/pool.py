"""
Pool cache for tabular Reveal-IG.

Precomputes, for each feature, the squared distances from the test point
to all pool members and the value-collapse structure needed for the
entropy path.
"""

from __future__ import annotations

import torch


class IdentityPoolCache:
    """
    Per-feature squared-distance structure for the entropy path.

    For each feature i:
      - test_sq_dists[i]: (M,) squared distances from x_test to each pool member
      - inverse[i]:       (M,) integer indices mapping pool rows to unique values
      - K[i]:             number of unique values of feature i in the pool
      - H_max[i]:         maximum entropy (uniform distribution over unique values)

    Distances are in the standardized input space (after zero-mean / unit-variance
    scaling), so features contribute proportionally to their empirical variability.

    Args:
        x_pool:           (M, D) tensor. Row test_idx_in_pool is the test point.
        test_idx_in_pool: Row index of the test point in x_pool (default 0).
    """

    def __init__(self, x_pool: torch.Tensor, test_idx_in_pool: int = 0) -> None:
        B, D = x_pool.shape
        self.M: int = B
        self.D: int = D
        self.test_idx: int = test_idx_in_pool
        self.x_pool: torch.Tensor = x_pool

        self.test_sq_dists: list[torch.Tensor] = []
        self.inverse: list[torch.Tensor] = []
        self.K: list[int] = []
        self.H_max: list[float] = []

        with torch.no_grad():
            for i in range(D):
                vals = x_pool[:, i]

                d_test = (vals - vals[test_idx_in_pool]).pow(2)
                self.test_sq_dists.append(d_test)

                unique_vals, inv = torch.unique(vals, return_inverse=True)
                K_i = unique_vals.numel()
                counts = torch.zeros(K_i, device=vals.device)
                counts.scatter_add_(0, inv, torch.ones_like(vals))
                p_v = counts / B
                H_max_i = -(p_v.clamp_min(1e-30) * p_v.clamp_min(1e-30).log()).sum().item()

                self.inverse.append(inv)
                self.K.append(K_i)
                self.H_max.append(H_max_i)
