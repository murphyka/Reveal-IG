# Reveal-IG: Attribution via Distributional Paths for Information Revelation

**Kieran A. Murphy and Shameen Shrestha** · New Jersey Institute of Technology

> **Paper:** *Attribution via Distributional Paths for Information Revelation* · [arXiv:2606.03885](https://arxiv.org/abs/2606.03885)

---

Reveal-IG is a feature attribution method that lifts path attribution from **input space** to **distribution space**. Instead of interpolating from a baseline input to the explained input (as in standard Integrated Gradients), Reveal-IG integrates gradients along a path through the space of probe distributions that progressively concentrate around the input. At each point on the path the model is queried under the current distribution via Monte Carlo sampling, and attribution is accumulated from the gradient of the expected model response with respect to the distribution parameters.

The result is a **complete, path-based attribution** -- attributions sum to the change in expected model output from the reference distribution to the endpoint -- that naturally handles multiscale image probes and feature-wise uncertainty in tabular data.

## Installation

```bash
pip install -e .
```

Requirements: Python $\ge$ 3.10, PyTorch $\ge$ 2.1, torchvision, Captum, NumPy, Matplotlib, tqdm, Pillow.

For the tabular demo, also install scikit-learn:

```bash
pip install -e ".[examples]"
```

## Quick start: images

Compare Reveal-IG against IG and SmoothGrad on two images:

```bash
python compare_images.py --images bird.jpg dog.jpg
```

`--adaptive-sigma` enables per-image adaptive stopping: a binary search finds the largest noise level the model tolerates while retaining $\ge$ 95% of its clean-image logit. 

With an ImageNet training directory for the Expected Gradients baseline:

```bash
python compare_images.py \
    --images path/to/image.jpg \
    --adaptive-sigma \
    --background-dir /path/to/imagenet/train/ \
    --n-background 100
```

The output is a PNG grid per image: top-15 predicted classes, then one attribution row per top-3 class showing Reveal-IG, IG, SmoothGrad, IDG, Guided IG, and Blur-IG.

Guided IG uses the [PAIR saliency library](https://github.com/PAIR-code/saliency); Blur-IG uses a custom scipy-based implementation. Both are included in `pip install "revealig[examples]"`.

### All `compare_images.py` options

| Flag | Default | Description |
|---|---|---|
| `--images` | *(required)* | Image file(s) or directory |
| `--outdir` | `results/` | Output directory |
| `--target` | argmax | ImageNet class index to attribute |
| `--n-steps` | 50 | Reveal-IG integration steps |
| `--n-samples` | 10 | Reveal-IG MC samples per step |
| `--ig-steps` | 50 | Steps for IG and IDG |
| `--sg-samples` | 50 | SmoothGrad samples |
| `--sigma-final` | 1/256 | Final σ (ignored with `--adaptive-sigma`) |
| `--adaptive-sigma` | off | Adaptive σ_stop |
| `--clip-pct` | 99 | Colour scale clip percentile |
| `--background-dir` | None | ImageNet train dir for Expected Gradients (optional) |
| `--n-background` | 100 | Number of background images |
| `--device` | auto | PyTorch device string |

## Quick start — tabular

Self-contained demo on California Housing regression (downloads automatically, no GPU needed, ~60 s on CPU):

```bash
python examples/tabular_demo.py
```

## Quick start — synthetic (2D toy functions)

No data or GPU needed — runs entirely on CPU in a few minutes:

```bash
python examples/synthetic_demo.py
```

Produces two figures in `results/`:
- `toy_heatmap_attributions.png` — ∇f, IG, Reveal-IG, and SHAP attribution maps side-by-side across 5 toy functions
- `multistep_shap.png` — multi-step SHAP (k=1,2,4,8) bridging standard SHAP and Reveal-IG on the same functions

```bash
python examples/synthetic_demo.py --only heatmap    # just the attribution comparison
python examples/synthetic_demo.py --only multistep  # just the multi-step figure
python examples/synthetic_demo.py --grid-resolution 60  # faster (default 100)
python examples/synthetic_demo.py --svg             # also save as SVG
```

Trains a small MLP, then runs Reveal-IG and KernelSHAP on a single test prediction and saves a side-by-side comparison to `tabular_attributions.png`. Features are standard-scored (zero mean, unit variance) so that pool distances are on a common scale across features.

## Python API

### Image attribution

```python
import torch
from torchvision.models import resnet50, ResNet50_Weights
from revealig.image.attribution import ImageAttributor
from revealig.image.stopping import find_sigma_stop

weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights).eval()

# x: (1, 3, 224, 224) ImageNet-normalised tensor
sigma = find_sigma_stop(model, x, target=243, tau=0.95)
attributor = ImageAttributor(model, n_steps=50, n_samples=10, sigma_final=sigma)
result = attributor.attribute(x, target=243, show_progress=True)

print(result)                    # ImageAttributionResult(shape=(3,224,224), ...)
print(result.completeness)       # sum(attr) ≈ E[f(x_final)] - E[f(x_noise)]
attr_2d = result.attr_map()      # (H, W) summed across channels (default)
attr_mu  = result.attr_mu        # (C, H, W) mean-shift component
attr_lv  = result.attr_logvar    # (C, H, W) variance-reduction component
```

### Tabular attribution

```python
import torch
from revealig.tabular import TabularAttributor

# X_train: (N, D) float tensor, standard-scored features (zero mean, unit variance).
# X_background: a representative subset of training data used as the reference pool.
# Standardize y as well so that E[f(background)] ≈ 0 and sum(attr) ≈ f(x_test).

attributor = TabularAttributor(model, n_steps=40, n_samples=40)
result = attributor.attribute(x_test, X_background, show_progress=True)

# result.attr: (D,) attributions in standardized output units
# sum(result.attr) ≈ E[f(x_test)] − E_{pool-uniform}[f(x)] ≈ f(x_test) when y is standardized
```

The tabular method integrates along an entropy path in pool-assignment space: at
each path position, feature i is sampled from the training pool using a softmax
with per-feature temperature τ_i(s), calibrated so that the value-collapsed
assignment entropy equals s · H_max_i. Gradients are computed in closed form
(no autograd in the integration loop). See the paper for details.

## Package layout

```
revealig/
  core/
    path.py         DistributionPath base + LinearPath
    integrator.py   RevealIG engine (model-agnostic)
    kl.py           KL divergence utilities
  image/
    attribution.py  ImageAttributor wrapper and ImageAttributionResult
    stopping.py     Adaptive sigma_stop via binary search
    viz.py          Attribution grid rendering
  tabular/
    pool.py         IdentityPoolCache (per-feature pool distance structure)
    integrator.py   TabularAttributor (pool-based entropy-path integration)
  compare/
    captum_baselines.py  Baseline attributors: IG, SmoothGrad, IDG, Guided IG, Blur-IG, Expected Gradients
compare_images.py          Image comparison CLI script
examples/
  tabular_demo.py   Standalone tabular attribution demo (California Housing)
  synthetic_demo.py Synthetic 2D toy testbed (no data needed; heatmap + multi-step SHAP figures)
```

## Citation

```bibtex
@article{murphy2026revealig,
  title   = {Attribution via Distributional Paths for Information Revelation},
  author  = {Murphy, Kieran A. and Shrestha, Shameen},
  journal = {arXiv preprint arXiv:2606.03885},
  year    = {2026},
}
```

## License

MIT
