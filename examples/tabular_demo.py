"""
Tabular Reveal-IG demo: California Housing regression.

Trains a small MLP on California Housing, then runs pool-based Reveal-IG and
KernelSHAP on a single test example and plots a side-by-side comparison.

Usage
-----
    pip install "revealig[examples]"   # adds scikit-learn and shap
    python examples/tabular_demo.py

No GPU required — runs fine on CPU in ~60 seconds.

Method notes
------------
Reveal-IG for tabular data integrates along an entropy path in pool-assignment
space.  At each point on the path, feature i is assigned by sampling from a
softmax distribution over a reference pool of training points, with a
per-feature temperature τ_i that decreases so the assignment concentrates
on pool members matching the test input.  Gradients are computed in closed
form (no autograd in the integration loop).

Features are standardized to zero mean / unit variance so that squared pool
distances are in a common scale across features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from revealig.tabular import TabularAttributor


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------

def load_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    housing = fetch_california_housing()
    X, y = housing.data.astype("float32"), housing.target.astype("float32")
    feature_names = list(housing.feature_names)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.1, random_state=42,
    )

    # Standard-score each feature to zero mean / unit variance so that pool
    # distances are on a common scale.  Unlike normalising to N(0,1) via
    # quantile transform, this keeps the original relative structure of
    # outliers, which is informative for attribution.
    sc_x = StandardScaler()
    X_tr = sc_x.fit_transform(X_tr).astype("float32")
    X_te = sc_x.transform(X_te).astype("float32")

    # Standard-score y so that E[f(background)] ≈ 0 and attribution sums
    # equal the model output directly (rather than model output − baseline).
    sc_y = StandardScaler()
    y_tr = sc_y.fit_transform(y_tr.reshape(-1, 1)).ravel().astype("float32")
    y_te = sc_y.transform(y_te.reshape(-1, 1)).ravel().astype("float32")

    return (
        torch.from_numpy(X_tr), torch.from_numpy(X_te),
        torch.from_numpy(y_tr), torch.from_numpy(y_te),
        feature_names,
    )


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------

def build_model(n_features: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(n_features, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


def train(
    model: nn.Module,
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    epochs: int = 800,
    lr: float = 1e-3,
    batch_size: int = 512,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = len(X_tr)
    model.train()
    for epoch in range(epochs):
        idx = torch.randperm(n)
        for start in range(0, n, batch_size):
            b = idx[start : start + batch_size]
            pred = model(X_tr[b]).squeeze(-1)
            loss = loss_fn(pred, y_tr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                train_rmse = loss_fn(model(X_tr).squeeze(-1), y_tr).sqrt().item()
            print(f"  epoch {epoch+1:4d}  train RMSE: {train_rmse:.4f}")
    model.eval()


# ---------------------------------------------------------------------------
# 3. Attribution
# ---------------------------------------------------------------------------

def attribute_reveal_ig(
    model: nn.Module,
    x: torch.Tensor,
    X_tr: torch.Tensor,
    pool_size: int = 256,
    n_steps: int = 60,
    n_samples: int = 100,
) -> torch.Tensor:
    """Pool-based Reveal-IG in standardized feature space. Returns (D,) attributions."""
    rng = np.random.default_rng(0)
    bg_idx = rng.choice(len(X_tr), size=pool_size, replace=False)
    X_bg = X_tr[bg_idx]

    attributor = TabularAttributor(model, n_steps=n_steps, n_samples=n_samples, s_start=0.99, s_end=0.05)
    result = attributor.attribute(x, X_bg, show_progress=True)

    print(f"  pool size:            {result.pool_size} (background {pool_size} + test point)")
    print(f"  sum(attr):            {result.attr.sum():.4f}")
    print(f"  model output on x:    {model(x.unsqueeze(0)).item():.4f}")
    return result.attr


def attribute_shap(
    model: nn.Module,
    x: torch.Tensor,
    X_tr: torch.Tensor,
    n_background: int = 50,
    n_samples: int = 512,
) -> torch.Tensor:
    """KernelSHAP. Returns (D,) Shapley values in standardized feature space."""
    import shap

    rng = np.random.default_rng(0)
    bg_idx = rng.choice(len(X_tr), size=n_background, replace=False)
    background = X_tr[bg_idx].numpy()

    def predict(x_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(x_np.astype("float32"))).squeeze(-1).numpy()

    explainer = shap.KernelExplainer(predict, background)
    shap_values = explainer.shap_values(x.numpy().reshape(1, -1), nsamples=n_samples)
    return torch.from_numpy(np.array(shap_values).squeeze().astype("float32"))


# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------

def plot_comparison(
    attr_reveal_ig: torch.Tensor,
    attr_shap: torch.Tensor,
    feature_names: list[str],
) -> None:
    a_reveal_ig = attr_reveal_ig.detach().cpu().numpy()
    a_shap = attr_shap.detach().cpu().numpy()

    order = np.argsort(np.abs(a_reveal_ig))
    names = [feature_names[i] for i in order]
    vals_reveal_ig = a_reveal_ig[order]
    vals_shap = a_shap[order]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, vals, title in [
        (axes[0], vals_reveal_ig, "Reveal-IG"),
        (axes[1], vals_shap, "KernelSHAP"),
    ]:
        colors = ["steelblue" if v > 0 else "tomato" for v in vals]
        ax.barh(range(len(names)), vals, color=colors)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Attribution (standardized output units)", fontsize=9)

    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=10)

    fig.suptitle(
        "Feature attributions — California Housing (sorted by |Reveal-IG|)", fontsize=11
    )
    fig.tight_layout()
    plt.savefig("results/tabular_attributions.png", dpi=150)
    print("\nSaved: results/tabular_attributions.png")
    plt.show()


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading California Housing...")
    X_tr, X_te, y_tr, y_te, feature_names = load_data()
    print(f"  train: {X_tr.shape}, test: {X_te.shape}")
    print(f"  features: {feature_names}")

    print("\nTraining MLP...")
    model = build_model(X_tr.shape[1])
    train(model, X_tr, y_tr, epochs=800)

    with torch.no_grad():
        test_rmse = nn.MSELoss()(model(X_te).squeeze(-1), y_te).sqrt().item()
    print(f"  test RMSE: {test_rmse:.4f}  (standardized units)")

    x = X_te[0]
    print(f"\nExample 0: y_true={y_te[0].item():.3f}, "
          f"y_pred={model(x.unsqueeze(0)).item():.3f}  (standardized units)")

    print("\nRunning Reveal-IG...")
    attr_reveal_ig = attribute_reveal_ig(model, x, X_tr)

    print("\nRunning KernelSHAP...")
    attr_shap = attribute_shap(model, x, X_tr)

    print("\nFeature attributions:")
    print(f"  {'Feature':<20}  {'Reveal-IG':>10}  {'KernelSHAP':>12}")
    print("  " + "-" * 46)
    for name, a, s in sorted(
        zip(feature_names, attr_reveal_ig.tolist(), attr_shap.tolist()),
        key=lambda t: -abs(t[1]),
    ):
        print(f"  {name:<20}  {a:>+10.4f}  {s:>+12.4f}")

    plot_comparison(attr_reveal_ig, attr_shap, feature_names)
