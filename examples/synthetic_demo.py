"""
Synthetic 2D attribution testbed — self-contained.

Produces two figures:
  results/toy_heatmap_attributions.png  — 5 functions × 5 columns (∇f, IG, REVEALIG, SHAP)
  results/multistep_shap.png            — SHAP → k-step SHAP (k=1,2,4,8) → REVEALIG

Quick start:
    python examples/synthetic_demo.py                    # both figures
    python examples/synthetic_demo.py --only heatmap
    python examples/synthetic_demo.py --only multistep
    python examples/synthetic_demo.py --grid-resolution 60   # faster
    python examples/synthetic_demo.py --svg              # also save SVG

Functions: xor, checkerboard, diagonal_ckb, radial, flat_far_field

Background distribution for all methods: uniform on [−EXTENT, EXTENT]².
SHAP and multi-step SHAP use a grid/MC approximation of this background.
REVEALIG implicitly integrates from this background toward a near-delta at the
query point.
"""

from __future__ import annotations

import argparse
import math
from itertools import combinations
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent.parent / "results"
OUT.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXTENT           = 2.0
GRID_RESOLUTION  = 100    # query-point grid for attributions (N×N)
HEAT_N           = 240    # function-value heatmap resolution
BG_PER_DIM       = 100    # exact-SHAP background grid per dimension

# REVEALIG parameters
REVEALIG_N_STEPS = 30
REVEALIG_N_MC    = 1365   # 30 × 1365 ≈ 41k grad evaluations per query point

# Multi-step SHAP parameters
MS_N_MC      = 4000   # MC samples per (state-pair, query-point) for multi-step SHAP
MS_REVEALIG_N_MC = 2600   # MC for REVEALIG in the multi-step figure
K_VALUES     = [1, 2, 4, 8]


# ── Toy functions ─────────────────────────────────────────────────────────

class XOR(nn.Module):
    """Smooth XOR: tanh(s·x) · tanh(s·y).  Origin is the saddle point."""
    def __init__(self, sharpness: float = 5.0):
        super().__init__()
        self.sharpness = sharpness

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(self.sharpness * x[..., 0])
                * torch.tanh(self.sharpness * x[..., 1]))


class Checkerboard(nn.Module):
    """cos·cos checkerboard, origin at the centre of a +1 square."""
    def __init__(self, period: float = 1.0, sharpness: float = 15.0):
        super().__init__()
        self.period    = period
        self.sharpness = sharpness

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = torch.cos(math.pi * x[..., 0] / self.period)
        v = torch.cos(math.pi * x[..., 1] / self.period)
        return torch.tanh(self.sharpness * u * v)


class DiagonalCheckerboard(nn.Module):
    """45°-rotated checkerboard: boundaries along x+y and x−y diagonals.

    Exposes axis-aligned methods (SHAP, IG): marginalising or integrating
    along one axis crosses the same number of sign-changes as the regular
    checkerboard, but the function is only separable in (x+y, x−y), not
    (x, y) — so axis-aligned attributions systematically mis-describe the
    interaction structure.
    """
    def __init__(self, period: float = 1.0, sharpness: float = 15.0):
        super().__init__()
        self.period    = period
        self.sharpness = sharpness

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = (x[..., 0] + x[..., 1]) / math.sqrt(2)
        v = (x[..., 0] - x[..., 1]) / math.sqrt(2)
        cu = torch.cos(math.pi * u / self.period)
        cv = torch.cos(math.pi * v / self.period)
        return torch.tanh(self.sharpness * cu * cv)


class RadialRings(nn.Module):
    """Cosine of radius, attenuated by a Gaussian envelope."""
    def __init__(self, k_rings: float = 1.0, env_sigma: float = 1.5):
        super().__init__()
        self.k         = k_rings
        self.env_sigma = env_sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r   = torch.norm(x, dim=-1)
        env = torch.exp(-r ** 2 / (2 * self.env_sigma ** 2))
        return env * torch.cos(2 * math.pi * self.k * r)


class FlatFarFieldBumps(nn.Module):
    """≈0 everywhere except narrow, high-amplitude Gaussian bumps near origin.

    IG shadow effect: a straight-line path from a far baseline to a far
    target that transits the bump cluster integrates large bump gradients
    and projects them onto the (target − baseline) direction, giving
    non-zero attribution for features that only describe far-field position.
    REVEALIG avoids this because its path stays in distribution space.
    """
    def __init__(self,
                 centers=((0.15, -0.10), (-0.12, 0.18), (0.05, 0.05)),
                 amplitudes=(1.0, -0.85, 0.70),
                 sigma: float = 0.12):
        super().__init__()
        self.register_buffer("centers",    torch.tensor(centers,    dtype=torch.float32))
        self.register_buffer("amplitudes", torch.tensor(amplitudes, dtype=torch.float32))
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff  = x.unsqueeze(-2) - self.centers   # (..., K, 2)
        r2    = (diff ** 2).sum(-1)               # (..., K)
        bumps = torch.exp(-r2 / (2 * self.sigma ** 2))
        return (bumps * self.amplitudes).sum(-1)


FUNCS: dict[str, nn.Module] = {
    "xor":            XOR(sharpness=5.0),
    "checkerboard":   Checkerboard(period=1.0, sharpness=15.0),
    "diagonal_ckb":   DiagonalCheckerboard(period=1.0, sharpness=15.0),
    "radial":         RadialRings(k_rings=1.0, env_sigma=1.5),
    "flat_far_field": FlatFarFieldBumps(),
}
FUNC_NAMES = list(FUNCS.keys())


# ── Grid helpers ──────────────────────────────────────────────────────────

def grid_points(n: int, extent: float = EXTENT) -> np.ndarray:
    xs = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    return np.stack([X.flatten(), Y.flatten()], axis=1).astype(np.float32)


def evaluate_heat(fn: nn.Module, n: int = HEAT_N, extent: float = EXTENT) -> np.ndarray:
    fn = fn.to(DEVICE)
    xs = torch.linspace(-extent, extent, n)
    X, Y = torch.meshgrid(xs, xs, indexing="xy")
    pts = torch.stack([X.flatten(), Y.flatten()], dim=1).to(DEVICE)
    with torch.no_grad():
        return fn(pts).cpu().numpy().reshape(n, n)


# ── Attribution methods ───────────────────────────────────────────────────

def gradient_field(fn: nn.Module, points: np.ndarray) -> np.ndarray:
    """∇f at all query points — single batched autograd call."""
    pts = torch.tensor(points, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    y = fn(pts)
    g = torch.autograd.grad(y.sum(), pts)[0]
    return g.detach().cpu().numpy()


def integrated_gradients(fn: nn.Module, points: np.ndarray,
                          n_steps: int = 64) -> np.ndarray:
    """IG from the (0, 0) baseline — midpoint rule, batched over query points."""
    pts      = torch.tensor(points, dtype=torch.float32, device=DEVICE)
    B, D     = pts.shape
    baseline = torch.zeros(D, device=DEVICE)
    delta    = pts - baseline
    attr     = torch.zeros(B, D, device=DEVICE)
    for k in range(n_steps):
        alpha = (k + 0.5) / n_steps
        x_a   = (baseline + alpha * delta).requires_grad_(True)
        y     = fn(x_a)
        g     = torch.autograd.grad(y.sum(), x_a)[0]
        attr += g.detach() * delta / n_steps
    return attr.detach().cpu().numpy()


def continuous_REVEALIG(fn: nn.Module, points: np.ndarray,
                    n_steps: int = REVEALIG_N_STEPS,
                    n_mc:    int = REVEALIG_N_MC,
                    extent:  float = EXTENT,
                    eps:     float = 0.02,
                    seed:    int   = 0) -> np.ndarray:
    """Continuous uniform-path REVEALIG, fully batched over all query points.

    Path (per query point x_q, independent per-feature uniform):
        μ_t  = t · x_q                            (centre: 0 → x_q)
        hw_t = extent·(1 − t) + eps·t             (half-width: EXTENT → eps)
    At t=0: U(−extent, extent)²  — broad background
    At t=1: U(x_q − eps, x_q + eps)²  — near-delta at x_q

    Attribution (chain rule through reparameterisation):
        attr_i = ∫₀¹ [ E[∂f/∂x_i] · x_q_i
                       + E[(2u_i−1)·∂f/∂x_i] · (eps−extent) ] dt

    Completeness: Σ attr_i ≈ f(x_q) − E_{U(−extent,extent)²}[f].
    """
    torch.manual_seed(seed)
    pts  = torch.tensor(points, dtype=torch.float32, device=DEVICE)
    B, D = pts.shape
    dt   = 1.0 / n_steps
    attr = torch.zeros(B, D, device=DEVICE)

    for step in range(n_steps):
        t    = (step + 0.5) * dt
        mu_t = t * pts
        hw_t = extent * (1.0 - t) + eps * t

        u      = torch.rand(B, n_mc, D, device=DEVICE)
        x_samp = mu_t.unsqueeze(1) + hw_t * (2.0 * u - 1.0)   # (B, n_mc, D)
        x_flat = x_samp.reshape(B * n_mc, D).requires_grad_(True)

        y     = fn(x_flat)
        grads = torch.autograd.grad(y.sum(), x_flat)[0]        # (B*n_mc, D)
        grads = grads.reshape(B, n_mc, D).detach()

        dE_dmu = grads.mean(1)                                  # (B, D)
        dE_dhw = ((2.0 * u.detach() - 1.0) * grads).mean(1)   # (B, D)

        attr += (dE_dmu * pts + dE_dhw * (eps - extent)) * dt

    return attr.detach().cpu().numpy()


def shap_2d(fn: nn.Module, points: np.ndarray,
            bg_n: int = BG_PER_DIM, extent: float = EXTENT) -> np.ndarray:
    """Exact Shapley values for 2 independent features, uniform background.

        SHAP_x = ½ [(g_x(x₁) − μ) + (f(x) − g_y(x₂))]
        SHAP_y = ½ [(g_y(x₂) − μ) + (f(x) − g_x(x₁))]

    where μ = E[f], g_x(x₁) = E_{X₂}[f(x₁, X₂)], g_y(x₂) = E_{X₁}[f(X₁, x₂)].
    Completeness: SHAP_x + SHAP_y = f(x) − μ.
    """
    bg_xs = torch.linspace(-extent, extent, bg_n, device=DEVICE)
    BX, BY = torch.meshgrid(bg_xs, bg_xs, indexing="xy")
    bg_pts = torch.stack([BX.flatten(), BY.flatten()], dim=1)
    with torch.no_grad():
        bg_vals = fn(bg_pts).reshape(bg_n, bg_n)
    mu       = bg_vals.mean()
    g_x_grid = bg_vals.mean(dim=0)   # marginalise y → function of x
    g_y_grid = bg_vals.mean(dim=1)   # marginalise x → function of y

    pts = torch.tensor(points, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        fx = fn(pts)

    def nearest(vals, grid):
        return torch.argmin((vals.unsqueeze(1) - grid.unsqueeze(0)).abs(), dim=1)

    g_x = g_x_grid[nearest(pts[:, 0], bg_xs)]
    g_y = g_y_grid[nearest(pts[:, 1], bg_xs)]

    shap_x = 0.5 * ((g_x - mu) + (fx - g_y))
    shap_y = 0.5 * ((g_y - mu) + (fx - g_x))
    return torch.stack([shap_x, shap_y], dim=1).cpu().numpy()


# ── Multi-step SHAP ───────────────────────────────────────────────────────

def _half_width(s: int, k: int, extent: float = EXTENT, eps: float = 0.0) -> float:
    """Half-width of the uniform distribution at step s/k (linear path)."""
    t = s / k
    return float(extent * (1.0 - t) + eps * t)


def _gen_valid_orderings(k: int) -> list[list[tuple[int, int]]]:
    """All C(2k, k) valid orderings of (2k) actions for 2 features with k steps each.

    An ordering is valid iff feature i's steps appear in increasing order.
    Equivalent to all ways to interleave two sorted sequences of length k.
    Action (feat, step): feat ∈ {0,1}, step ∈ {1,...,k}.
    """
    total = 2 * k
    orderings: list[list[tuple[int, int]]] = []
    for pos_f0 in combinations(range(total), k):
        pos_set = set(pos_f0)
        pos_f1  = [i for i in range(total) if i not in pos_set]
        ordering: list[tuple[int, int]] = [None] * total  # type: ignore[list-item]
        for step_idx, pos in enumerate(pos_f0):
            ordering[pos] = (0, step_idx + 1)
        for step_idx, pos in enumerate(pos_f1):
            ordering[pos] = (1, step_idx + 1)
        orderings.append(ordering)
    return orderings


def _compute_E_table(
    fn: nn.Module,
    pts: torch.Tensor,      # (B, 2)
    k: int,
    n_mc: int = MS_N_MC,
    seed: int = 0,
) -> dict[tuple[int, int], torch.Tensor]:
    """E[f | feat0 at step a, feat1 at step b] for all (a,b) in {0,...,k}².

    Feature i at step s:
        center_s = (s/k) · x_i
        hw_s     = _half_width(s, k)   — linear path, same as REVEALIG

    At s=0: U[−extent, extent] — global background.
    At s=k: U[x_i−eps, x_i+eps] — near-delta at x_i.
    k=1 recovers standard Shapley values exactly (up to MC noise).
    """
    torch.manual_seed(seed)
    B = pts.shape[0]
    E: dict[tuple[int, int], torch.Tensor] = {}
    for a in range(k + 1):
        hw_a  = _half_width(a, k)
        ctr_a = (a / k) * pts[:, 0:1]   # (B, 1)
        for b in range(k + 1):
            hw_b  = _half_width(b, k)
            ctr_b = (b / k) * pts[:, 1:2]
            u1 = torch.rand(B, n_mc, device=pts.device)
            u2 = torch.rand(B, n_mc, device=pts.device)
            x1 = ctr_a + hw_a * (2 * u1 - 1)   # (B, n_mc)
            x2 = ctr_b + hw_b * (2 * u2 - 1)
            xy = torch.stack([x1, x2], dim=2).reshape(B * n_mc, 2)
            with torch.no_grad():
                f_vals = fn(xy).reshape(B, n_mc)
            E[(a, b)] = f_vals.mean(dim=1)       # (B,)
    return E


def multistep_shap_2d(
    fn: nn.Module,
    points: np.ndarray,
    k: int = 1,
    n_mc: int = MS_N_MC,
    seed: int = 0,
) -> np.ndarray:
    """k-step Shapley values for 2D with independent uniform background.

    k=1 recovers standard Shapley values exactly (up to MC noise).
    k→∞ converges to REVEALIG along the same linear distribution path.
    Returns (N, 2) attribution array.
    """
    pts = torch.tensor(points, dtype=torch.float32, device=DEVICE)
    B   = pts.shape[0]

    E         = _compute_E_table(fn, pts, k, n_mc=n_mc, seed=seed)
    orderings = _gen_valid_orderings(k)
    n_ord     = len(orderings)   # == C(2k, k)

    attr = torch.zeros(B, 2, device=DEVICE)
    for ordering in orderings:
        state = [0, 0]
        for feat, step in ordering:
            s_before = (state[0], state[1])
            state[feat] = step
            s_after  = (state[0], state[1])
            attr[:, feat] += (E[s_after] - E[s_before]) / n_ord

    return attr.cpu().numpy()


# ── Figure 1: heatmap comparison ──────────────────────────────────────────

HEATMAP_METHODS = [
    ("∇f",   "grad"),
    ("IG",   "ig"),
    ("REVEALIG", "REVEALIG"),
    ("SHAP", "shap"),
]


def compute_heatmap(n: int = GRID_RESOLUTION) -> dict[str, np.ndarray]:
    points = grid_points(n)
    data: dict[str, np.ndarray] = {}
    for name, fn in FUNCS.items():
        fn = fn.to(DEVICE)
        print(f"[heatmap/{name}] {n}×{n} grid ...", flush=True)
        t0 = time()
        data[f"{name}_heat"] = evaluate_heat(fn)
        data[f"{name}_grad"] = gradient_field(fn, points)
        data[f"{name}_ig"]   = integrated_gradients(fn, points)
        data[f"{name}_REVEALIG"] = continuous_REVEALIG(fn, points, eps=0.02)
        data[f"{name}_shap"] = shap_2d(fn, points, bg_n=n)
        print(f"  done in {time() - t0:.1f}s")
    return data


def render_heatmap(data: dict[str, np.ndarray],
                   out_path: Path,
                   n: int = GRID_RESOLUTION,
                   extra_formats: tuple = ()) -> None:
    nrows = len(FUNC_NAMES)
    ncols = 1 + len(HEATMAP_METHODS)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.6 * ncols, 2.6 * nrows + 0.3))
    if nrows == 1:
        axes = axes.reshape(1, ncols)

    for ri, name in enumerate(FUNC_NAMES):
        heat = data[f"{name}_heat"]
        vmax = max(1e-6, float(np.abs(heat).max()))

        ax = axes[ri, 0]
        im = ax.imshow(heat, extent=[-EXTENT, EXTENT, -EXTENT, EXTENT],
                       origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.set_ylabel(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title("f", fontsize=11)

        for ci, (mname, key) in enumerate(HEATMAP_METHODS, start=1):
            attrs = data[f"{name}_{key}"]
            diff  = (attrs[:, 0] - attrs[:, 1]).reshape(n, n)
            ref   = max(float(np.abs(diff).max()), 0.1)
            ax    = axes[ri, ci]
            ax.imshow(diff, extent=[-EXTENT, EXTENT, -EXTENT, EXTENT],
                      origin="lower", cmap="PuOr", vmin=-ref, vmax=ref)
            if ri == 0:
                ax.set_title(mname, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.02, 0.97, f"max={np.abs(diff).max():.2f}",
                    transform=ax.transAxes, fontsize=7, va="top",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              alpha=0.7, edgecolor="none"))

    fig.suptitle(
        f"A_x − A_y attribution diff  ({n}×{n} grid; "
        f"background = Unif[−{EXTENT}, {EXTENT}]²)",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    for fmt in extra_formats:
        alt = out_path.with_suffix(f".{fmt}")
        fig.savefig(alt, bbox_inches="tight")
        print(f"Saved {alt}")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Figure 2: multi-step SHAP ─────────────────────────────────────────────

def compute_multistep(n: int = GRID_RESOLUTION,
                      k_values: list[int] = K_VALUES,
                      n_mc: int = MS_N_MC) -> dict[str, np.ndarray]:
    points = grid_points(n)
    data: dict[str, np.ndarray] = {}
    for name, fn in FUNCS.items():
        fn = fn.to(DEVICE)
        print(f"[multistep/{name}]", flush=True)
        t0 = time()
        data[f"{name}_heat"] = evaluate_heat(fn)

        t1 = time()
        data[f"{name}_shap"] = shap_2d(fn, points, bg_n=n)
        print(f"  SHAP (exact)  {time()-t1:.1f}s")

        for k in k_values:
            n_ord = math.comb(2 * k, k)
            t1 = time()
            data[f"{name}_k{k}"] = multistep_shap_2d(fn, points, k=k, n_mc=n_mc)
            print(f"  k={k}  C(2k,k)={n_ord}  {time()-t1:.1f}s")

        t1 = time()
        data[f"{name}_REVEALIG"] = continuous_REVEALIG(fn, points,
                                               n_mc=MS_REVEALIG_N_MC, eps=0.0)
        print(f"  REVEALIG  {time()-t1:.1f}s")
        print(f"  total {time()-t0:.1f}s")
    return data


def _attr_panel(ax, attrs, n, title, ri):
    diff = (attrs[:, 0] - attrs[:, 1]).reshape(n, n)
    ref  = max(float(np.abs(diff).max()), 1e-6)
    ax.imshow(diff, extent=[-EXTENT, EXTENT, -EXTENT, EXTENT],
              origin="lower", cmap="PuOr", vmin=-ref, vmax=ref)
    ax.set_xticks([]); ax.set_yticks([])
    if ri == 0:
        ax.set_title(title, fontsize=11)
    ax.text(0.02, 0.97, f"max={np.abs(diff).max():.2f}",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      alpha=0.7, edgecolor="none"))


def render_multistep(data: dict[str, np.ndarray],
                     out_path: Path,
                     n: int = GRID_RESOLUTION,
                     k_values: list[int] = K_VALUES,
                     extra_formats: tuple[str, ...] = ()) -> None:
    # columns: f | SHAP | k=1 | k=2 | … | REVEALIG
    ncols = 1 + 1 + len(k_values) + 1
    nrows = len(FUNC_NAMES)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.6 * ncols, 2.6 * nrows + 0.5))
    if nrows == 1:
        axes = axes.reshape(1, ncols)

    for ri, name in enumerate(FUNC_NAMES):
        heat = data[f"{name}_heat"]
        vmax = max(1e-6, float(np.abs(heat).max()))

        ax = axes[ri, 0]
        im = ax.imshow(heat, extent=[-EXTENT, EXTENT, -EXTENT, EXTENT],
                       origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.set_ylabel(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title("f", fontsize=11)

        _attr_panel(axes[ri, 1], data[f"{name}_shap"], n,
                    "SHAP\n(exact)", ri)

        for ci, k in enumerate(k_values, start=2):
            n_ord = math.comb(2 * k, k)
            _attr_panel(axes[ri, ci], data[f"{name}_k{k}"], n,
                        f"k={k}  ({n_ord} ord.)", ri)

        _attr_panel(axes[ri, -1], data[f"{name}_REVEALIG"], n,
                    "REVEALIG\n(continuous)", ri)

    k_max = k_values[-1]
    hw_vals = " → ".join(
        f"{_half_width(s, k_max):.3f}" for s in range(k_max + 1)
    )
    fig.suptitle(
        f"Multi-step SHAP  —  A_x − A_y  ({n}×{n} grid, {MS_N_MC} MC/state)\n"
        f"Half-widths at k={k_max}: {hw_vals}  "
        f"[linear: extent·(1−s/k), extent={EXTENT}]",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.0, 1, 0.95])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    for fmt in extra_formats:
        alt = out_path.with_suffix(f".{fmt}")
        fig.savefig(alt, bbox_inches="tight")
        print(f"Saved {alt}")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=["heatmap", "multistep"],
                   help="Produce only one figure (default: both)")
    p.add_argument("--grid-resolution", type=int, default=GRID_RESOLUTION,
                   help=f"Query grid N×N (default {GRID_RESOLUTION})")
    p.add_argument("--k", type=int, nargs="+", default=K_VALUES,
                   help="k values for multi-step SHAP (default: 1 2 4 8)")
    p.add_argument("--n-mc", type=int, default=MS_N_MC,
                   help=f"MC samples per state pair (default {MS_N_MC})")
    p.add_argument("--svg", action="store_true", help="Also save as SVG")
    args = p.parse_args()

    n              = args.grid_resolution
    extra_formats  = ("svg",) if args.svg else ()

    if args.only != "multistep":
        data = compute_heatmap(n=n)
        render_heatmap(data,
                       out_path=OUT / "toy_heatmap_attributions.png",
                       n=n,
                       extra_formats=extra_formats)

    if args.only != "heatmap":
        data = compute_multistep(n=n, k_values=args.k, n_mc=args.n_mc)
        render_multistep(data,
                         out_path=OUT / "multistep_shap.png",
                         n=n,
                         k_values=args.k,
                         extra_formats=extra_formats)


if __name__ == "__main__":
    main()
