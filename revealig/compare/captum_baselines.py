"""
Baseline attribution methods for comparison with Reveal-IG.

All functions return a (H, W) attribution tensor (channels summed) to match
the channel-reduction used in the paper's visualisations.

Methods provided:
  - run_ig:             Integrated Gradients (Captum), zero baseline
  - run_smoothgrad:     SmoothGrad (Captum NoiseTunnel wrapping Saliency)
  - run_idg:            Integrated Decision Gradients (Walker et al., AAAI 2024)
  - run_guided_ig:      Guided IG (Kapishnikov et al., CVPR 2021) via PAIR saliency lib
  - run_blur_ig:        Blur-IG (Xu et al., CVPR 2020) — custom midpoint implementation
  - run_expected_gradients: Expected Gradients (Captum GradientShap),
                            baseline = random background images (optional).

run_guided_ig requires: pip install saliency
run_blur_ig requires: pip install scipy
"""

from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn

from captum.attr import GradientShap, IntegratedGradients, NoiseTunnel, Saliency


def _channel_sum(attr: torch.Tensor) -> torch.Tensor:
    """(1, C, H, W) or (C, H, W) -> (H, W) by summing across channels."""
    if attr.dim() == 4:
        attr = attr.squeeze(0)
    return attr.sum(dim=0)


# ── Captum baselines ──────────────────────────────────────────────────────

def run_ig(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    n_steps: int = 50,
    baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrated Gradients (Captum), zero baseline. Returns (H, W)."""
    model.eval()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(next(model.parameters()).device)
    if baseline is None:
        baseline = torch.zeros_like(x)
    else:
        baseline = baseline.to(x.device)
        if baseline.dim() == 3:
            baseline = baseline.unsqueeze(0)

    attr = IntegratedGradients(model).attribute(
        x, baselines=baseline, target=target, n_steps=n_steps, method="gausslegendre",
    )
    return _channel_sum(attr.detach())


def run_smoothgrad(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    n_samples: int = 50,
    stdev_spread: float = 0.15,
) -> torch.Tensor:
    """SmoothGrad (Captum NoiseTunnel wrapping Saliency). Returns (H, W)."""
    model.eval()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(next(model.parameters()).device).requires_grad_(True)
    input_range = float((x.max() - x.min()).item())

    attr = NoiseTunnel(Saliency(model)).attribute(
        x,
        nt_type="smoothgrad",
        nt_samples=n_samples,
        stdevs=stdev_spread * input_range,
        target=target,
        abs=False,
    )
    return _channel_sum(attr.detach())


def run_expected_gradients(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    background: torch.Tensor,
    n_samples: int = 50,
) -> torch.Tensor:
    """Expected Gradients (Captum GradientShap, stdevs=0). Returns (H, W)."""
    model.eval()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(next(model.parameters()).device)
    background = background.to(x.device)

    attr = GradientShap(model).attribute(
        x, baselines=background, target=target, n_samples=n_samples, stdevs=0.0,
    )
    return _channel_sum(attr.detach())


# ── IDG ───────────────────────────────────────────────────────────────────

def run_idg(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    n_steps: int = 50,
) -> torch.Tensor:
    """
    Integrated Decision Gradients (Walker et al., AAAI 2024).

    Faithful port of github.com/chasewalker26/Integrated-Decision-Gradients.
    Phase 1 pre-characterises the decision path with uniform forward passes,
    then Phase 2 redistributes integration density to high-slope regions.
    Returns (H, W).
    """
    model.eval()
    dev = next(model.parameters()).device
    xb  = (x if x.dim() == 4 else x.unsqueeze(0)).to(dev).detach()
    N   = n_steps
    step_size = 1.0 / N

    # Phase 1: uniform forward passes to estimate logit slopes
    alphas_u = torch.linspace(0, 1, N, device=dev).reshape(N, 1, 1, 1)
    with torch.no_grad():
        logits_u = model(alphas_u * xb)[:, target]   # (N,)

    slopes = torch.zeros(N, device=dev)
    slopes[1:] = (logits_u[1:] - logits_u[:-1]) / step_size

    # Phase 2a: redistribute alpha positions by slope (public IDG logic)
    s_min, s_max = slopes.min(), slopes.max()
    slopes_01 = (slopes - s_min) / (s_max - s_min + 1e-12)
    slopes_01[0] = 0.0
    slopes_norm = slopes_01 / (slopes_01.sum() + 1e-12)
    sample_f = slopes_norm * N
    sample_i = sample_f.int().clone()
    remaining = int(N - sample_i.sum().item())
    sample_f_copy = sample_f.clone()
    sample_f_copy[sample_i != 0] = -1.0
    fill_order = torch.flip(torch.sort(sample_f_copy)[1], dims=[0])
    sample_i[fill_order[:remaining]] = 1

    new_alphas    = torch.zeros(N, device=dev)
    substep_sizes = torch.zeros(N, device=dev)
    idx, val = 0, 0.0
    for n_sub_t in sample_i:
        n_sub = int(n_sub_t.item())
        if n_sub == 0:
            continue   # public code: does NOT advance val
        region = torch.linspace(val, val + step_size, n_sub + 1, device=dev)[:n_sub]
        new_alphas[idx: idx + n_sub]    = region
        substep_sizes[idx: idx + n_sub] = step_size / n_sub
        idx += n_sub
        val += step_size

    # Phase 2b: batched gradients at redistributed alpha positions
    interp = (new_alphas.reshape(N, 1, 1, 1) * xb).requires_grad_(True)
    out    = model(interp)[:, target]
    grads  = torch.autograd.grad(out, interp, grad_outputs=torch.ones_like(out))[0].detach()
    out    = out.detach()

    new_slopes = torch.zeros(N, device=dev)
    dalphas = new_alphas[1:] - new_alphas[:-1]
    valid   = dalphas.abs() > 1e-7
    new_slopes[1:][valid] = (out[1:] - out[:-1])[valid] / dalphas[valid]

    weighted = grads * new_slopes.view(N, 1, 1, 1) * substep_sizes.view(N, 1, 1, 1)
    return _channel_sum((weighted.mean(dim=0) * xb.squeeze(0)).detach())


# ── PAIR saliency library baselines ──────────────────────────────────────

def run_guided_ig(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
) -> torch.Tensor:
    """
    Guided IG (Kapishnikov et al., CVPR 2021) via the PAIR saliency library.

    Requires: pip install saliency
    Returns (H, W).
    """
    import saliency.core as sc

    model.eval()
    dev = next(model.parameters()).device
    x0  = (x if x.dim() == 4 else x.unsqueeze(0)).to(dev).detach()
    x_np = x0[0].cpu().numpy()   # (C, H, W) float32

    def call_model_function(x_batch, call_model_args=None, expected_keys=None):
        t = torch.tensor(x_batch, dtype=torch.float32, device=dev).requires_grad_(True)
        grads = torch.autograd.grad(model(t)[:, target].sum(), t)[0]
        return {sc.INPUT_OUTPUT_GRADIENTS: grads.detach().cpu().numpy()}

    attr_np = sc.GuidedIG().GetMask(x_np, call_model_function)   # (C, H, W) float64
    return _channel_sum(torch.from_numpy(attr_np.astype(np.float32)))


def run_blur_ig(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    n_steps: int = 50,
    sigma_max: float = 10.0,
) -> torch.Tensor:
    """
    Blur-IG (Xu et al., CVPR 2020).

    Midpoint-rule integration of ∇f(blur(x, σ)) from σ=sigma_max to σ≈0,
    weighted by (x − blur(x, sigma_max)).  Returns (H, W).
    """
    from scipy.ndimage import gaussian_filter

    model.eval()
    dev  = next(model.parameters()).device
    xb   = (x if x.dim() == 4 else x.unsqueeze(0)).to(dev)
    x_np = xb[0].detach().cpu().numpy()   # (C, H, W)
    integrated = torch.zeros(x_np.shape, device=dev)

    for k in range(n_steps):
        alpha   = (k + 0.5) / n_steps
        sigma_k = sigma_max * (1.0 - alpha)
        blurred = (
            np.stack([gaussian_filter(x_np[c], sigma=sigma_k) for c in range(x_np.shape[0])])
            if sigma_k > 0.01 else x_np.copy()
        )
        xt = torch.tensor(blurred, dtype=xb.dtype, device=dev).unsqueeze(0).requires_grad_(True)
        model(xt)[0, target].backward()
        if xt.grad is not None:
            integrated += xt.grad[0].detach()

    baseline = np.stack([gaussian_filter(x_np[c], sigma=sigma_max) for c in range(x_np.shape[0])])
    bl_t = torch.tensor(baseline, dtype=xb.dtype, device=dev)
    return _channel_sum(((xb[0] - bl_t) * integrated / n_steps).detach())


# ── Convenience entry point ───────────────────────────────────────────────

def run_all(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    ig_steps: int = 50,
    sg_samples: int = 50,
    background: torch.Tensor | None = None,
    eg_samples: int = 50,
) -> dict[str, torch.Tensor]:
    """
    Run all baselines and return a dict of (H, W) attribution maps.

    Always runs: IG, SmoothGrad, IDG, Guided IG, Blur-IG.
    Also runs Expected Gradients if background is provided.
    Guided IG and Blur-IG require `pip install saliency`.
    """
    def _run(name: str, fn):
        print(f"    {name} ...", end="", flush=True)
        t0 = time.time()
        result = fn()
        print(f" {time.time() - t0:.1f}s")
        return result

    results: dict[str, torch.Tensor] = {
        "IG":         _run("IG",         lambda: run_ig(model, x, target=target, n_steps=ig_steps)),
        "SmoothGrad": _run("SmoothGrad", lambda: run_smoothgrad(model, x, target=target, n_samples=sg_samples)),
        "IDG":        _run("IDG",        lambda: run_idg(model, x, target=target, n_steps=ig_steps)),
        "Guided IG":  _run("Guided IG",  lambda: run_guided_ig(model, x, target=target)),
        "Blur-IG":    _run("Blur-IG",    lambda: run_blur_ig(model, x, target=target)),
    }
    if background is not None:
        results["Exp. Gradients"] = _run(
            "Exp. Gradients",
            lambda: run_expected_gradients(model, x, target=target, background=background, n_samples=eg_samples),
        )
    return results
