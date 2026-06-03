"""
Visualisation utilities for attribution maps.

Primary entry point: attribution_grid(), which renders:
  - Left: top-K probability bar chart + original image (both spanning all rows)
  - Right: attribution columns (Reveal-IG, Captum baselines) with one row per class
"""

from __future__ import annotations

from typing import Any

import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def _hwc_image(x: torch.Tensor) -> np.ndarray:
    """(C, H, W) tensor -> (H, W, C) normalised float in [0, 1] for display."""
    img = _to_numpy(x).transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def _attr_to_rgb(
    attr: torch.Tensor,
    clip_percentile: float = 99.0,
    cmap: str = "RdBu_r",
) -> np.ndarray:
    """(H, W) attribution -> (H, W, 3) RGB. Negative -> red, positive -> blue."""
    a = _to_numpy(attr)
    clip = np.percentile(np.abs(a), clip_percentile)
    if clip == 0:
        clip = 1e-8
    a_norm = np.clip(a / clip, -1.0, 1.0)
    a_01 = (a_norm + 1.0) / 2.0
    return cm.get_cmap(cmap)(a_01)[:, :, :3]


def show_attribution(
    attr_map: torch.Tensor,
    title: str = "",
    ax: Any | None = None,
    clip_percentile: float = 99.0,
    cmap: str = "RdBu_r",
) -> Any:
    """Display a (H, W) attribution map as a standalone heatmap."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(_attr_to_rgb(attr_map, clip_percentile=clip_percentile, cmap=cmap))
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    return ax


def attribution_grid(
    image: torch.Tensor,
    top_k_probs: list[tuple[str, float]],
    rows: list[dict],
    clip_percentile: float = 99.0,
    figsize_per_panel: tuple[float, float] = (3.0, 3.2),
    probs_panel_width: float = 4.0,
    orig_panel_width: float = 3.0,
) -> Figure:
    """
    Render a multi-row attribution comparison grid.

    Layout
    ------
    Left (spanning all rows):
      col 0 -- top-K probability bar chart
      col 1 -- original image

    Right (one row per class):
      col 2 -- Reveal-IG
      col 3+ -- one per entry in rows[0]["captum"]

    Args:
        image:            (C, H, W) input tensor.
        top_k_probs:      List of (class_name, probability) for the top-K classes.
        rows:             List of dicts, one per class to show:
                            {
                              "label":  str,
                              "reveal_ig":   ImageAttributionResult,
                              "captum": dict[str, (H, W) Tensor],
                            }
        clip_percentile:  Colour scale clip percentile for all attribution panels.
    """
    n_rows = len(rows)
    captum_names = list(rows[0]["captum"].keys())
    n_attr_cols = 1 + len(captum_names)
    n_total_cols = 2 + n_attr_cols

    attr_w = figsize_per_panel[0]
    panel_h = figsize_per_panel[1]

    col_widths = (
        [probs_panel_width, orig_panel_width]
        + [attr_w] * n_attr_cols
    )
    total_w = sum(col_widths)
    total_h = panel_h * n_rows

    fig = plt.figure(figsize=(total_w, total_h))
    gs = gridspec.GridSpec(
        n_rows, n_total_cols,
        figure=fig,
        width_ratios=col_widths,
        hspace=0.35,
        wspace=0.08,
    )

    # Left col 0: top-K probability bar chart (spans all rows)
    ax_probs = fig.add_subplot(gs[:, 0])
    _render_probs_bar(ax_probs, top_k_probs)

    # Left col 1: original image (spans all rows)
    ax_orig = fig.add_subplot(gs[:, 1])
    ax_orig.imshow(_hwc_image(image))
    ax_orig.set_title("Input", fontsize=8)
    ax_orig.axis("off")

    # Attribution columns, one row per class
    attr_col_titles = ["Reveal-IG"] + captum_names

    for r, row in enumerate(rows):
        label = row["label"]
        reveal_ig = row["reveal_ig"]
        captum = row["captum"]

        attr_maps = [reveal_ig.attr_map("sum")] + [captum[name] for name in captum_names]

        for c, (title, amap) in enumerate(zip(attr_col_titles, attr_maps)):
            ax = fig.add_subplot(gs[r, 2 + c])
            row_label = f"\n[{label}]" if c == 0 else ""
            col_title = title if r == 0 else ""
            ax_title = f"{col_title}{row_label}".strip()
            show_attribution(amap, title=ax_title, ax=ax, clip_percentile=clip_percentile)

    return fig


def _render_probs_bar(ax: Any, top_k_probs: list[tuple[str, float]]) -> None:
    """Horizontal bar chart of top-K class probabilities."""
    names = [_truncate(name, 28) for name, _ in top_k_probs]
    probs = [p for _, p in top_k_probs]

    y = list(range(len(names)))
    bars = ax.barh(y, probs, align="center", color="steelblue", height=0.7)

    bars[0].set_color("tomato")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Probability", fontsize=7)
    ax.set_title("Top predictions", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlim(0, max(probs) * 1.25)

    for bar, p in zip(bars, probs):
        ax.text(
            bar.get_width() + max(probs) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{p:.1%}",
            va="center",
            fontsize=6,
        )


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "..."
