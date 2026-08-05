#!/usr/bin/env python3
"""
Side-by-side comparison: ground truth | restored | |GT - restored|.

Matches restored PNGs named like:
  radio_train_packed_calibration_target_4_proj0000_ground_truth_restored.png
to ground truths under the radiography data root.

Usage:
  python compare_restorations.py \\
      --restored-dir save_folder/radio_1/model-3750 \\
      --data-root ../../radiography/fix_test_realistic_v2

  python compare_restorations.py --restored-dir save_folder/radio_1/model-3750 --show
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

try:
    import imageio.v3 as imageio
except ImportError:
    import imageio

import matplotlib.pyplot as plt


RESTORED_RE = re.compile(
    r"^(?P<split>radio_train|radio_test)_(?P<stem>.+)_restored\.png$"
)


def load_gray(path: Path) -> np.ndarray:
    img = np.asarray(imageio.imread(path))
    if img.ndim == 3:
        # RGBA/RGB -> luminance (or first channel if already gray-ish)
        if img.shape[2] >= 3:
            rgb = img[:, :, :3].astype(np.float32)
            img = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
        else:
            img = img[:, :, 0].astype(np.float32)
    else:
        img = img.astype(np.float32)
    return img


def find_ground_truth(data_root: Path, stem: str) -> Path | None:
    """stem e.g. packed_calibration_target_4_proj0000_ground_truth"""
    # Prefer nested folder layout used by fix_test_realistic_v2
    # packed_calibration_target_4/packed_calibration_target_4_proj0000_ground_truth.png
    m = re.match(r"(packed_calibration_target_\d+)_", stem)
    if m:
        nested = data_root / m.group(1) / f"{stem}.png"
        if nested.is_file():
            return nested
    # Flat fallback / recursive search
    direct = data_root / f"{stem}.png"
    if direct.is_file():
        return direct
    hits = list(data_root.rglob(f"{stem}.png"))
    return hits[0] if hits else None


def make_panel(gt: np.ndarray, restored: np.ndarray) -> tuple[np.ndarray, dict]:
    if gt.shape != restored.shape:
        raise ValueError(f"shape mismatch GT {gt.shape} vs restored {restored.shape}")

    # Work in [0, 1] for display consistency
    gt_f = gt / 255.0 if gt.max() > 1.5 else gt
    rest_f = restored / 255.0 if restored.max() > 1.5 else restored
    diff = np.abs(gt_f - rest_f)

    mse = float(np.mean((gt_f - rest_f) ** 2))
    psnr = 100.0 if mse < 1e-12 else float(20.0 * np.log10(1.0 / np.sqrt(mse)))
    metrics = {
        "mse": mse,
        "psnr": psnr,
        "diff_mean": float(diff.mean()),
        "diff_max": float(diff.max()),
    }
    return diff, metrics


def plot_triplet(
    gt: np.ndarray,
    restored: np.ndarray,
    diff: np.ndarray,
    title: str,
    metrics: dict,
    out_path: Path | None,
    show: bool,
    diff_vmax: float | None,
):
    gt_f = gt / 255.0 if gt.max() > 1.5 else gt
    rest_f = restored / 255.0 if restored.max() > 1.5 else restored
    vmax = diff_vmax if diff_vmax is not None else max(diff.max(), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gt_f, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Ground truth")
    axes[0].axis("off")

    axes[1].imshow(rest_f, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Restored")
    axes[1].axis("off")

    im = axes[2].imshow(diff, cmap="magma", vmin=0, vmax=vmax)
    axes[2].set_title("|GT − restored|")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{title}\n"
        f"PSNR={metrics['psnr']:.2f} dB | "
        f"mean|diff|={metrics['diff_mean']:.4f} | "
        f"max|diff|={metrics['diff_max']:.4f}",
        fontsize=11,
    )
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Wrote {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--restored-dir",
        type=Path,
        default=Path("save_folder/radio_1/model-3750"),
        help="Directory containing *_restored.png (and optionally model-*.pt)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("../../radiography/fix_test_realistic_v2"),
        help="Root containing ground-truth PNGs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write comparison panels (default: <restored-dir>/comparisons)",
    )
    parser.add_argument(
        "--split",
        choices=["all", "radio_train", "radio_test"],
        default="all",
        help="Which restored split(s) to include",
    )
    parser.add_argument(
        "--diff-vmax",
        type=float,
        default=None,
        help="Fixed color scale max for |diff| (default: per-image max)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Interactively display figures (in addition to saving)",
    )
    args = parser.parse_args()

    restored_dir = args.restored_dir.resolve()
    data_root = args.data_root.resolve()
    output_dir = (args.output_dir or (restored_dir / "comparisons")).resolve()

    if not restored_dir.is_dir():
        print(f"Error: restored-dir not found: {restored_dir}")
        return 1
    if not data_root.is_dir():
        print(f"Error: data-root not found: {data_root}")
        return 1

    restored_files = sorted(restored_dir.glob("*_restored.png"))
    if not restored_files:
        print(f"Error: no *_restored.png in {restored_dir}")
        return 1

    n_ok = 0
    for path in restored_files:
        m = RESTORED_RE.match(path.name)
        if not m:
            print(f"Skip (name pattern): {path.name}")
            continue
        split = m.group("split")
        stem = m.group("stem")
        if args.split != "all" and split != args.split:
            continue

        gt_path = find_ground_truth(data_root, stem)
        if gt_path is None:
            print(f"Missing GT for {path.name} (stem={stem})")
            continue

        gt = load_gray(gt_path)
        restored = load_gray(path)
        # Crop restored to GT size if padding leftovers remain
        h, w = gt.shape[:2]
        restored = restored[:h, :w]
        if restored.shape != gt.shape:
            print(f"Skip shape mismatch: {path.name} {restored.shape} vs {gt_path} {gt.shape}")
            continue

        diff, metrics = make_panel(gt, restored)
        out_path = output_dir / f"{split}_{stem}_compare.png"
        plot_triplet(
            gt, restored, diff,
            title=f"{split} | {stem}",
            metrics=metrics,
            out_path=out_path,
            show=args.show,
            diff_vmax=args.diff_vmax,
        )
        n_ok += 1

    print(f"Done: {n_ok} comparison panel(s) -> {output_dir}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
