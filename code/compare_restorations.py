#!/usr/bin/env python3
"""Comparison plots for {train|test|eval}_{id}_restored.png.

Run from code/. Looks up matching ground truth (and noisy input) under --data-root.

Layouts:

  2x2  (default)
    GT | corrupted
    restored | |GT − restored|
      python compare_restorations.py \\
        --restored-dir save_folder/radio_4/model-10000 \\
        --data-root ../data-generation/fix_test_realistic_v2

  3x2  (--error-attenuation)
    GT                  | rescale-only error
    rescaled corrupted  | restored error
    restored            | error attenuation
      python compare_restorations.py --error-attenuation \\
        --restored-dir save_folder/radio_4/model-10000 \\
        --data-root ../data-generation/fix_test_realistic_v2

  1x2  (--error-attenuation --layout 1x2)
    GT | error attenuation
      python compare_restorations.py --error-attenuation --layout 1x2 \\
        --restored-dir save_folder/radio_4/model-10000 \\
        --data-root ../data-generation/fix_test_realistic_v2

Writes {split}_{id}_compare.png, or {split}_{id}_compare_1x2.png for the 1x2 layout.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import imageio.v3 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from skimage.metrics import structural_similarity

RESTORED_RE = re.compile(r"^(?P<split>train|test|eval)_(?P<id>.+?)_restored\.png$")


def load_gray(path: Path) -> np.ndarray:
    img = np.load(path) if path.suffix.lower() == ".npy" else np.asarray(imageio.imread(path))
    if img.ndim == 3:
        rgb = img[:, :, :3].astype(np.float32)
        return 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
    return img.astype(np.float32)


def to_unit(img: np.ndarray) -> np.ndarray:
    """Map integer / float images into [0, 1] via dtype or magnitude peak (display helper)."""
    arr = np.asarray(img)
    if arr.size == 0:
        return arr.astype(np.float32)
    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr.astype(np.float32) / float(np.iinfo(arr.dtype).max), 0.0, 1.0)
    arr = arr.astype(np.float32, copy=False)
    mx = float(np.nanmax(arr)) if arr.size else 1.0
    if mx <= 1.0 + 1e-3:
        peak = 1.0
    elif mx <= 255.0 + 1e-3:
        peak = 255.0
    else:
        peak = 65535.0
    return np.clip(arr / peak, 0.0, 1.0)


def data_range(gt: np.ndarray) -> float:
    """Intensity span of the ground truth (max − min)."""
    gt = np.asarray(gt, dtype=np.float64)
    dr = float(gt.max() - gt.min())
    return dr if dr > 0 else 1.0


def find_radiograph(data_root: Path, target_id: str, kind: str, ext: str = ".png") -> Path | None:
    """Find a GT or noisy PNG/NPY for a calibration-target id or STL/run stem."""
    if not ext.startswith("."):
        ext = f".{ext}"
    suffix = f"_{kind}{ext}"

    folder = data_root / f"packed_calibration_target_{target_id}"
    hits = sorted(folder.glob(f"*{suffix}"))
    if hits:
        return hits[0]

    direct = sorted(data_root.glob(f"{target_id}_proj*{suffix}"))
    if direct:
        return direct[0]

    nested_dir = data_root / target_id
    if nested_dir.is_dir():
        nested = sorted(nested_dir.glob(f"{target_id}_proj*{suffix}"))
        if nested:
            return nested[0]
        nested = sorted(nested_dir.glob(f"*{suffix}"))
        if nested:
            return nested[0]

    fallback = data_root / f"packed_calibration_target_{target_id}_proj0000{suffix}"
    return fallback if fallback.is_file() else None


def find_gt(data_root: Path, target_id: str, ext: str = ".png") -> Path | None:
    return find_radiograph(data_root, target_id, "ground_truth", ext=ext)


def diff_norm(diff: np.ndarray, vmax: float | None, log_scale: bool):
    vmax = float(vmax) if vmax is not None else float(max(diff.max(), 1e-12))
    if not log_scale:
        return diff, Normalize(vmin=0.0, vmax=max(vmax, 1e-12))
    positive = diff[diff > 0]
    vmin = 1e-6 if positive.size == 0 else max(float(np.percentile(positive, 1.0)), vmax * 1e-6, 1e-8)
    if vmin >= vmax:
        vmax = vmin * 10.0
    return np.clip(diff.astype(np.float64), vmin, None), LogNorm(vmin=vmin, vmax=vmax)


def metrics_and_diff(gt: np.ndarray, restored: np.ndarray):
    """Absolute error / GT data range; PSNR and SSIM use the same data_range."""
    gt_f = np.asarray(gt, dtype=np.float64)
    rest_f = np.asarray(restored, dtype=np.float64)
    dr = data_range(gt_f)
    err = np.abs(gt_f - rest_f)
    diff = err / dr
    mse = float(np.mean(err ** 2))
    psnr = 100.0 if mse < 1e-12 else float(20.0 * np.log10(dr / np.sqrt(mse)))
    side = int(min(gt_f.shape[:2]))
    win = min(7, side if side % 2 == 1 else max(side - 1, 1))
    if win >= 3:
        ssim = float(structural_similarity(gt_f, rest_f, data_range=dr, win_size=win))
    else:
        ssim = 1.0 if mse < 1e-12 else 0.0
    return diff, {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "data_range": dr,
        "diff_mean": float(diff.mean()),
        "diff_max": float(diff.max()),
    }


def intensity_peaks(
    img: np.ndarray,
    n_grid: int = 1024,
    n_sample: int = 40_000,
    bw_method: float = 0.08,
    min_sep_frac: float = 0.08,
) -> tuple[float, float]:
    """Two KDE modes of the intensity histogram, sorted low to high.

    Endpoints are eligible (open-beam and dark floor often sit there).
    If a second well-separated mode is missing, fall back to (min, max).
    """
    x = np.asarray(img, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (0.0, 0.0)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo:
        return (lo, lo)
    if x.size > n_sample:
        step = max(1, x.size // n_sample)
        sample = x[::step][:n_sample]
    else:
        sample = x
    if np.unique(sample).size < 2:
        return (lo, hi)
    kde = gaussian_kde(sample, bw_method=bw_method)
    grid = np.linspace(lo, hi, n_grid)
    dens = kde(grid)
    cand = []
    if dens[0] >= dens[1]:
        cand.append(0)
    cand.extend(int(i) for i in find_peaks(dens)[0])
    if dens[-1] >= dens[-2]:
        cand.append(n_grid - 1)
    cand = sorted(set(cand), key=lambda i: dens[i], reverse=True)
    min_sep = min_sep_frac * (hi - lo)
    chosen: list[float] = []
    for i in cand:
        xi = float(grid[i])
        if all(abs(xi - c) >= min_sep for c in chosen):
            chosen.append(xi)
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        return (lo, hi)
    return (min(chosen), max(chosen))


def match_range(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Affine-map img onto ref so the two KDE intensity peaks coincide."""
    img = np.asarray(img, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    p_i = intensity_peaks(img)
    p_r = intensity_peaks(ref)
    denom = p_i[1] - p_i[0]
    if denom <= 0:
        return np.full_like(img, 0.5 * (p_r[0] + p_r[1]))
    return (img - p_i[0]) / denom * (p_r[1] - p_r[0]) + p_r[0]


def intensity_quantum(img: np.ndarray) -> float:
    """Smallest intensity step treated as nonzero (1 ADU, or 1/65535 in unit floats)."""
    _, peak = display_limits(img)
    return 1.0 if peak > 1.0 else max(peak / 65535.0, 1e-12)


def error_attenuation_map(
    gt: np.ndarray, corrupted: np.ndarray, restored: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """|GT − restored| / |GT − range-matched corrupted|, floored at 1 intensity count.

    Lower is better. Returns (attenuation, corrupted abs. error, intensity quantum).
    """
    gt_f = np.asarray(gt, dtype=np.float64)
    corr = match_range(corrupted, gt_f)
    rest_f = np.asarray(restored, dtype=np.float64)
    err_in = np.abs(gt_f - corr)
    err_out = np.abs(gt_f - rest_f)
    quantum = intensity_quantum(gt_f)
    att = err_out / np.maximum(err_in, quantum)
    return att, err_in, quantum


def l2_error_ratio(gt: np.ndarray, corrupted: np.ndarray, restored: np.ndarray) -> float:
    """‖GT − restored‖₂ / ‖GT − range-matched corrupted‖₂ (lower is better)."""
    gt_f = np.asarray(gt, dtype=np.float64)
    corr = match_range(corrupted, gt_f)
    rest_f = np.asarray(restored, dtype=np.float64)
    denom = float(np.linalg.norm(gt_f - corr))
    numer = float(np.linalg.norm(gt_f - rest_f))
    if denom <= 0:
        return 0.0 if numer <= 0 else float("inf")
    return numer / denom


def rescaled_corrupted_diff(gt: np.ndarray, corrupted: np.ndarray):
    """Abs. error of KDE-peak-rescaled corrupted vs GT, divided by GT data range."""
    return metrics_and_diff(gt, match_range(corrupted, gt))


def attenuation_norm(att: np.ndarray, err_in: np.ndarray, quantum: float, log_scale: bool):
    """Color limits from well-defined ratios (corrupted error above 1 count)."""
    reliable = att[(err_in > quantum) & np.isfinite(att) & (att > 0)]
    if reliable.size == 0:
        reliable = att[np.isfinite(att) & (att > 0)]
    if reliable.size == 0:
        return att, Normalize(vmin=0.0, vmax=1.0)
    vmin = float(np.percentile(reliable, 1.0))
    vmax = float(np.percentile(reliable, 99.0))
    vmin = min(max(vmin, 1e-3), 1.0)
    vmax = max(vmax, 1.0)
    if vmin >= vmax:
        vmax = vmin * 10.0
    if log_scale:
        return np.clip(att.astype(np.float64), vmin, None), LogNorm(vmin=vmin, vmax=vmax)
    return att, Normalize(vmin=0.0, vmax=vmax)


def display_limits(img: np.ndarray) -> tuple[float, float]:
    """Absolute intensity limits for display (no per-image contrast stretch)."""
    arr = np.asarray(img)
    if np.issubdtype(arr.dtype, np.integer):
        return 0.0, float(np.iinfo(arr.dtype).max)
    mx = float(np.nanmax(arr)) if arr.size else 1.0
    if mx <= 1.0 + 1e-3:
        return 0.0, 1.0
    if mx <= 255.0 + 1e-3:
        return 0.0, 255.0
    return 0.0, 65535.0


def _panel_layout(h: int, w: int, n_panels: int = 1, nrows: int | None = None, ncols: int | None = None):
    """Figure size / fonts scaled so panels stay near native resolution."""
    dpi = 100
    panel_in = h / dpi
    scale = panel_in / 5.0
    if nrows is None or ncols is None:
        nrows, ncols = 1, n_panels
    fs_panel = max(12.0, 14.0 * scale)
    n_gaps = max(nrows - 1, 0)
    caption_in = (fs_panel / 72.0) + 0.35 * scale if n_gaps else 0.0
    extra_h = n_gaps * caption_in + (0.5 * scale if nrows > 1 else 0.0)
    return {
        "dpi": dpi,
        "scale": scale,
        "fs_panel": fs_panel,
        "fs_sup": max(14.0, 16.0 * scale),
        "fs_cbar": max(10.0, 12.0 * scale),
        "caption_in": caption_in,
        "panel_in": panel_in,
        # Extra width for a free-standing colorbar to the right of the error panel.
        "fig_w": ncols * (w / dpi) + 2.0 * scale,
        "fig_h": nrows * panel_in + 1.5 * scale + extra_h,
    }


def _savefig(fig, out_path: Path, dpi: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        format="png",
        pil_kwargs={"compress_level": 6},
    )
    print(f"Wrote {out_path}")


def plot_triplet(
    gt,
    restored,
    diff,
    title,
    metrics,
    out_path,
    show,
    diff_vmax,
    log_diff,
    corrupted=None,
    error_attenuation=False,
    layout="3x2",
    show_header=True,
):
    gt_f = np.asarray(gt, dtype=np.float64)
    rest_f = np.asarray(restored, dtype=np.float64)
    vmin, vmax = display_limits(gt)
    h, w = gt_f.shape[:2]
    att = None
    use_att = bool(error_attenuation and corrupted is not None)
    layout_1x2 = bool(use_att and layout == "1x2")
    # Display order is row-major. Heat maps stay in the right column so
    # their colorbars stack in parallel (3x2). 1x2 is GT | attenuation.
    if layout_1x2:
        att, att_err_in, att_q = error_attenuation_map(gt_f, corrupted, rest_f)
        att_plot, att_cmap_norm = attenuation_norm(att, att_err_in, att_q, log_diff)
        panels = [
            ("gray", "Ground truth", gt_f, None, None, (vmin, vmax)),
            ("heat", "Error attenuation (lower = better)", att_plot, att_cmap_norm, "restored error / baseline error", None),
        ]
        nrows, ncols = 1, 2
        show_header = False
    elif use_att:
        corr_scaled = match_range(corrupted, gt_f)
        rescale_diff, _ = metrics_and_diff(gt_f, corr_scaled)
        shared_vmax = diff_vmax
        if shared_vmax is None:
            shared_vmax = float(max(float(rescale_diff.max()), float(np.max(diff)), 1e-12))
        stacked = np.concatenate(
            [np.asarray(rescale_diff, dtype=np.float64).ravel(), np.asarray(diff, dtype=np.float64).ravel()]
        )
        _, err_norm = diff_norm(stacked, shared_vmax, log_diff)
        vmin_err = float(err_norm.vmin)
        if log_diff:
            rescale_plot = np.clip(np.asarray(rescale_diff, dtype=np.float64), vmin_err, None)
            diff_plot = np.clip(np.asarray(diff, dtype=np.float64), vmin_err, None)
        else:
            rescale_plot = rescale_diff
            diff_plot = diff
        att, att_err_in, att_q = error_attenuation_map(gt_f, corrupted, rest_f)
        att_plot, att_cmap_norm = attenuation_norm(att, att_err_in, att_q, log_diff)
        gray_lim = (vmin, vmax)
        panels = [
            ("gray", "Ground truth", gt_f, None, None, gray_lim),
            ("heat", "Rescale-only error", rescale_plot, err_norm, "abs. error / data range", None),
            ("gray", "Rescaled corrupted", corr_scaled, None, None, gray_lim),
            ("heat", "Restored error", diff_plot, err_norm, "abs. error / data range", None),
            ("gray", "Restored", rest_f, None, None, gray_lim),
            ("heat", "Error attenuation (lower = better)", att_plot, att_cmap_norm, "restored error / baseline error", None),
        ]
        nrows, ncols = 3, 2
    else:
        diff_plot, err_norm = diff_norm(diff, diff_vmax, log_diff)
        gray_lim = (vmin, vmax)
        panels = [("gray", "Ground truth", gt_f, None, None, gray_lim)]
        if corrupted is not None:
            corr_raw = np.asarray(corrupted, dtype=np.float64)
            panels.append(("gray", "Corrupted", corr_raw, None, None, display_limits(corr_raw)))
        panels.append(("gray", "Restored", rest_f, None, None, gray_lim))
        panels.append(("heat", "Error", diff_plot, err_norm, "abs. error / data range", None))
        n_panels = len(panels)
        nrows, ncols = (2, 2) if n_panels == 4 else (1, n_panels)

    layout_info = _panel_layout(h, w, n_panels=len(panels), nrows=nrows, ncols=ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(layout_info["fig_w"], layout_info["fig_h"]), dpi=layout_info["dpi"]
    )
    ax_list = np.atleast_1d(axes).ravel()
    heat_ims = []
    for ax, (kind, _, img, heat_norm, _, gray_lim) in zip(ax_list, panels):
        if kind == "gray":
            gvmin, gvmax = gray_lim if gray_lim is not None else (vmin, vmax)
            ax.imshow(img, cmap="gray", vmin=gvmin, vmax=gvmax, interpolation="nearest")
        else:
            heat_ims.append(ax.imshow(img, cmap="jet", norm=heat_norm, interpolation="nearest"))
        ax.axis("off")

    extra = ""
    if att is not None:
        extra = f" | L2 err ratio={l2_error_ratio(gt_f, corrupted, rest_f):.3g}"
    top_rect = 1.0
    if show_header:
        fig.suptitle(
            f"{title}\n"
            f"PSNR={metrics['psnr']:.2f} dB | "
            f"SSIM={metrics['ssim']:.4f}"
            f"{extra}",
            fontsize=layout_info["fs_sup"],
            y=0.98,
        )
        top_rect = 0.90
    hspace = (
        layout_info["caption_in"] / layout_info["panel_in"]
        if nrows > 1 and layout_info["panel_in"] > 0
        else 0.0
    )
    fig.tight_layout(rect=(0, 0, 0.90, top_rect))
    fig.subplots_adjust(hspace=hspace)

    heat_axes = [ax for ax, (kind, *_) in zip(ax_list, panels) if kind == "heat"]
    heat_labels = [cbar_label for kind, _, _, _, cbar_label, _ in panels if kind == "heat"]
    if use_att and not layout_1x2:
        top, mid, att_ax = heat_axes
        top_pos = top.get_position()
        mid_pos = mid.get_position()
        shared_cax = fig.add_axes(
            [top_pos.x1 + 0.02, mid_pos.y0, 0.018, top_pos.y1 - mid_pos.y0]
        )
        shared_cbar = fig.colorbar(heat_ims[0], cax=shared_cax)
        shared_cbar.ax.tick_params(labelsize=layout_info["fs_cbar"])
        shared_cbar.set_label(heat_labels[0], fontsize=layout_info["fs_cbar"])
        att_pos = att_ax.get_position()
        att_cax = fig.add_axes([att_pos.x1 + 0.02, att_pos.y0, 0.018, att_pos.height])
        att_cbar = fig.colorbar(heat_ims[2], cax=att_cax)
        att_cbar.ax.tick_params(labelsize=layout_info["fs_cbar"])
        att_cbar.set_label(heat_labels[2], fontsize=layout_info["fs_cbar"])
    else:
        for im, ax, cbar_label in zip(heat_ims, heat_axes, heat_labels):
            pos = ax.get_position()
            cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.018, pos.height])
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=layout_info["fs_cbar"])
            cbar.set_label(cbar_label, fontsize=layout_info["fs_cbar"])

    for ax, (_, label, *_) in zip(ax_list, panels):
        p = ax.get_position()
        fig.text(
            p.x0 + p.width / 2.0,
            p.y1 + 0.012,
            label,
            ha="center",
            va="bottom",
            fontsize=layout_info["fs_panel"],
        )

    if out_path is not None:
        _savefig(fig, out_path, layout_info["dpi"])
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_error(diff, out_path, show, diff_vmax, log_diff):
    """Standalone error map with labeled colorbar (no figure caption)."""
    diff_plot, norm = diff_norm(diff, diff_vmax, log_diff)
    h, w = diff.shape[:2]
    layout = _panel_layout(h, w, n_panels=1)

    fig, ax = plt.subplots(figsize=(layout["fig_w"], layout["fig_h"]), dpi=layout["dpi"])
    im = ax.imshow(diff_plot, cmap="jet", norm=norm, interpolation="nearest")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.08)
    cbar.ax.tick_params(labelsize=layout["fs_cbar"])
    cbar.set_label("abs. error / data range", fontsize=layout["fs_cbar"])
    fig.tight_layout()
    if out_path is not None:
        _savefig(fig, out_path, layout["dpi"])
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--restored-dir",
        type=Path,
        default=Path("save_folder/radio_4/model-10000"),
        help="Folder of {train|test|eval}_{id}_restored.png from train.py or eval_stl.py",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("../data-generation/fix_test_realistic_v2"),
        help="Dataset root used to look up matching *_ground_truth and *_noisy files",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write comparison PNGs (default: <restored-dir>/comparisons)",
    )
    p.add_argument(
        "--split",
        choices=["all", "train", "test", "eval"],
        default="all",
        help="Only plot this split's restorations (default: all)",
    )
    p.add_argument(
        "--diff-vmax",
        type=float,
        default=None,
        help="Colorbar cap for abs. error / data range (default: max error in the figure)",
    )
    p.add_argument(
        "--linear-diff",
        action="store_true",
        help="Linear error color scale (default is log)",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Open each comparison in a window instead of only writing PNGs",
    )
    p.add_argument(
        "--gt-ext",
        choices=["png", "npy"],
        default="png",
        help="Extension of the ground-truth / noisy files under --data-root",
    )
    p.add_argument(
        "--error-attenuation",
        action="store_true",
        help="Replace the default 2x2 with error attenuation layout (3x2, or 1x2 if --layout 1x2)",
    )
    p.add_argument(
        "--layout",
        choices=["3x2", "1x2"],
        default="3x2",
        help="Only with --error-attenuation: 3x2 (GT / rescaled / restored + errors) or 1x2 (GT | attenuation)",
    )
    args = p.parse_args()

    restored_dir = args.restored_dir.resolve()
    data_root = args.data_root.resolve()
    output_dir = (args.output_dir or restored_dir / "comparisons").resolve()
    if not restored_dir.is_dir():
        print(f"Error: restored-dir not found: {restored_dir}")
        return 1
    if not data_root.is_dir():
        print(f"Error: data-root not found: {data_root}")
        return 1

    files = sorted(restored_dir.glob("*_restored.png"))
    if not files:
        print(f"Error: no *_restored.png in {restored_dir}")
        return 1

    n_ok = 0
    for path in files:
        m = RESTORED_RE.match(path.name)
        if not m:
            print(f"Skip (name pattern): {path.name}")
            continue
        split, tid = m.group("split"), m.group("id")
        if args.split != "all" and split != args.split:
            continue

        gt_path = find_gt(data_root, tid, ext=args.gt_ext)
        if gt_path is None:
            print(f"Missing GT for {path.name}")
            continue
        noisy_path = find_radiograph(data_root, tid, "noisy", ext=args.gt_ext)

        gt = load_gray(gt_path)
        restored = load_gray(path)[: gt.shape[0], : gt.shape[1]]
        if restored.shape != gt.shape:
            print(f"Skip shape mismatch: {path.name}")
            continue
        corrupted = None
        if noisy_path is not None:
            noisy = load_gray(noisy_path)[: gt.shape[0], : gt.shape[1]]
            if noisy.shape == gt.shape:
                corrupted = noisy
            else:
                print(f"Skip corrupted (shape mismatch): {noisy_path.name}")
        else:
            print(f"Missing noisy for {path.name}; compare without corrupted panel")

        if args.error_attenuation and corrupted is None:
            print(f"Skip error-attenuation for {path.name} (no noisy image); using default layout")

        diff, metrics = metrics_and_diff(gt, restored)
        title = f"{split} | target {tid}"
        print(
            f"Pair {path.name} <- {gt_path.relative_to(data_root)} "
            f"(PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f})"
        )
        compare_name = (
            f"{split}_{tid}_compare_1x2.png"
            if args.error_attenuation and args.layout == "1x2"
            else f"{split}_{tid}_compare.png"
        )
        plot_triplet(
            gt, restored, diff,
            title=title,
            metrics=metrics,
            out_path=output_dir / compare_name,
            show=args.show,
            diff_vmax=args.diff_vmax,
            log_diff=not args.linear_diff,
            corrupted=corrupted,
            error_attenuation=args.error_attenuation,
            layout=args.layout,
        )
        if args.layout == "1x2":
            n_ok += 1
            continue
        plot_error(
            diff,
            out_path=output_dir / f"{split}_{tid}_error.png",
            show=False,
            diff_vmax=args.diff_vmax,
            log_diff=not args.linear_diff,
        )
        if corrupted is not None:
            rescale_diff, _ = rescaled_corrupted_diff(gt, corrupted)
            plot_error(
                rescale_diff,
                out_path=output_dir / f"{split}_{tid}_rescale_only_error.png",
                show=False,
                diff_vmax=args.diff_vmax,
                log_diff=not args.linear_diff,
            )
        n_ok += 1

    print(f"Done: {n_ok} pair(s) -> {output_dir}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
