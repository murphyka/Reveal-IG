"""
Compare Reveal-IG attribution against baselines on a single image or directory.

Runs Reveal-IG, IG, SmoothGrad, IDG, Guided IG, and Blur-IG on each image and
saves a side-by-side PNG grid.  Expected Gradients is also included if
--background-dir is supplied.

Guided IG and Blur-IG require: pip install "revealig[examples]"

Usage
-----
    # Single image (auto-detects target class)
    python compare_images.py --images path/to/image.jpg --outdir results/

    # Directory of images
    python compare_images.py --images path/to/imgs/ --outdir results/

    # With adaptive sigma_stop
    python compare_images.py --images image.jpg --outdir results/ --adaptive-sigma

    # With Expected Gradients baseline (needs ImageNet train dir for background)
    python compare_images.py --images image.jpg --outdir results/ \
        --background-dir /path/to/imagenet/train/ --n-background 100

    # Override target class
    python compare_images.py --images image.jpg --target 243 --outdir results/
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

from revealig.compare.captum_baselines import run_all
from revealig.image.attribution import ImageAttributor
from revealig.image.stopping import find_sigma_stop
from revealig.image.viz import attribution_grid

# -------------------------------------------------------------------
# ImageNet preprocessing (matches ResNet50_Weights.IMAGENET1K_V2)
# -------------------------------------------------------------------
IMAGENET_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_imagenet_classes() -> list[str]:
    weights = ResNet50_Weights.IMAGENET1K_V2
    return weights.meta["categories"]


def _load_image(path: Path) -> torch.Tensor:
    """Load and preprocess a single image -> (1, 3, 224, 224)."""
    img = Image.open(path).convert("RGB")
    return IMAGENET_TRANSFORM(img).unsqueeze(0)


def _load_background(
    train_dir: Path,
    n_background: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Load n_background random images from an ImageNet-style train directory
    for use as Expected Gradients baselines.

    Expects structure: train_dir/synset_folder/image.JPEG
    """
    all_images: list[Path] = []
    exts = {".jpg", ".jpeg", ".png", ".JPEG"}
    for synset_dir in train_dir.iterdir():
        if synset_dir.is_dir():
            for f in synset_dir.iterdir():
                if f.suffix in exts:
                    all_images.append(f)

    if len(all_images) < n_background:
        print(f"[warn] Only found {len(all_images)} images in {train_dir}, "
              f"requested {n_background}")
        n_background = len(all_images)

    chosen = random.sample(all_images, n_background)
    tensors = []
    for p in chosen:
        try:
            tensors.append(_load_image(p).squeeze(0))
        except Exception:
            pass

    return torch.stack(tensors).to(device)


def _collect_image_paths(paths: list[str]) -> list[Path]:
    """Expand paths: files stay as-is, directories are globbed for images."""
    result: list[Path] = []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            for ext in exts:
                result.extend(sorted(pp.glob(f"*{ext}")))
                result.extend(sorted(pp.glob(f"*{ext.upper()}")))
        elif pp.is_file():
            result.append(pp)
        else:
            print(f"[warn] Path not found, skipping: {p}", file=sys.stderr)
    return result


def compare(
    image_paths: list[Path],
    outdir: Path,
    target: int | None,
    n_steps: int,
    n_samples: int,
    ig_steps: int,
    sg_samples: int,
    sigma_final: float,
    adaptive_sigma: bool,
    clip_pct: float,
    device: torch.device,
    background: torch.Tensor | None = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Build model
    print("[compare] Loading ResNet50 (IMAGENET1K_V2)...")
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights).to(device).eval()
    classes = _load_imagenet_classes()

    for img_path in image_paths:
        print(f"\n[compare] Processing: {img_path.name}")
        try:
            x = _load_image(img_path).to(device)  # (1, 3, 224, 224)

            # Forward pass: logits -> top-15 probs and top-3 class indices
            with torch.no_grad():
                logits = model(x)
            probs = logits.softmax(dim=-1)[0]

            top15_probs, top15_idx = probs.topk(15)
            top_k_probs = [
                (classes[int(i)], float(p))
                for i, p in zip(top15_idx, top15_probs)
            ]

            # Select target classes for attribution rows
            if target is not None:
                top3_idx = [target] + [
                    int(i) for i in top15_idx[:4] if int(i) != target
                ][:2]
            else:
                top3_idx = [int(i) for i in top15_idx[:3]]

            print("  top-3 targets:")
            for cls_i in top3_idx:
                print(f"    [{cls_i}] {classes[cls_i]}: {float(probs[cls_i]):.1%}")

            # Determine sigma_final (adaptive or fixed)
            sf = sigma_final
            if adaptive_sigma:
                primary_target = top3_idx[0]
                sf = find_sigma_stop(model, x, target=primary_target, tau=0.95)
                sf = max(sf, 1.0 / 256.0)  # floor at 1/256
                print(f"  adaptive sigma_stop = {sf:.6f}")

            # Build attributor with the chosen sigma_final
            attributor = ImageAttributor(
                model,
                n_steps=n_steps,
                n_samples=n_samples,
                sigma_final=sf,
                device=device,
            )

            # Attribution for each of the top-3 classes
            rows = []
            for cls_idx in top3_idx:
                cls_label = classes[cls_idx]
                print(f"  attributing class {cls_idx} ({cls_label})...")
                print(f"    Reveal-IG ...", end="", flush=True)
                t0 = time.time()
                reveal_ig_result = attributor.attribute(x, target=cls_idx, show_progress=False)
                print(f" {time.time() - t0:.1f}s")
                captum_maps = run_all(
                    model, x, target=cls_idx,
                    ig_steps=ig_steps, sg_samples=sg_samples,
                    background=background,
                )
                rows.append({"label": cls_label, "reveal_ig": reveal_ig_result, "captum": captum_maps})

            # Render grid
            fig = attribution_grid(
                image=x.squeeze(0),
                top_k_probs=top_k_probs,
                rows=rows,
                clip_percentile=clip_pct,
            )

            out_path = outdir / f"{img_path.stem}_attribution.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved -> {out_path}")

        except Exception as exc:
            print(f"[compare] ERROR on {img_path.name}: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reveal-IG vs baseline attribution comparison grid",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images", nargs="+", required=True,
        help="Image file(s) or director(ies) to process",
    )
    parser.add_argument("--outdir", default="results/", help="Output directory")
    parser.add_argument(
        "--target", type=int, default=None,
        help="ImageNet class index to attribute (default: argmax of model prediction)",
    )
    parser.add_argument("--n-steps", type=int, default=50, help="Reveal-IG integration steps")
    parser.add_argument("--n-samples", type=int, default=10, help="Reveal-IG MC samples per step")
    parser.add_argument("--ig-steps", type=int, default=50, help="Captum IG integration steps")
    parser.add_argument("--sg-samples", type=int, default=50, help="Captum SmoothGrad samples")
    parser.add_argument(
        "--sigma-final", type=float, default=0.25,
        help="Reveal-IG final distribution stddev (ignored if --adaptive-sigma is set)",
    )
    parser.add_argument(
        "--adaptive-sigma", action="store_true",
        help="Use adaptive sigma_stop (binary search for max noise the model tolerates).",
    )
    parser.add_argument("--clip-pct", type=float, default=99.0, help="Colour scale clip percentile")
    parser.add_argument(
        "--background-dir", type=str, default=None,
        help="Path to ImageNet train directory for Expected Gradients baseline. "
             "If not provided, Expected Gradients is skipped.",
    )
    parser.add_argument(
        "--n-background", type=int, default=100,
        help="Number of background images for Expected Gradients",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device string (e.g. 'cuda', 'cpu'). Default: auto.",
    )

    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[compare] Using device: {device}")

    image_paths = _collect_image_paths(args.images)
    if not image_paths:
        print("[compare] No images found. Exiting.", file=sys.stderr)
        sys.exit(1)
    print(f"[compare] Found {len(image_paths)} image(s).")

    # Load background for Expected Gradients if requested
    background = None
    if args.background_dir is not None:
        print(f"[compare] Loading {args.n_background} background images...")
        background = _load_background(
            Path(args.background_dir), args.n_background, device,
        )
        print(f"[compare] Background shape: {background.shape}")

    compare(
        image_paths=image_paths,
        outdir=Path(args.outdir),
        target=args.target,
        n_steps=args.n_steps,
        n_samples=args.n_samples,
        ig_steps=args.ig_steps,
        sg_samples=args.sg_samples,
        sigma_final=args.sigma_final,
        adaptive_sigma=args.adaptive_sigma,
        clip_pct=args.clip_pct,
        device=device,
        background=background,
    )


if __name__ == "__main__":
    main()
