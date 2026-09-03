#!/usr/bin/env python3
"""Simulate a new STL with radiograph.py, then evaluate a trained RDBM checkpoint.

Example:
  python eval_stl.py ../data-generation/conebeam_abel_stepwedge_fixed.stl \\
      --checkpoint save_folder/radio_4/model-10000/model-10000.pt

  python eval_stl.py path/a.stl path/b.stl --skip-sim   # reuse existing radiographs
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

# Local imports (run from RDBM_radio/code/)
from datasets_setting import set_seed
from networks import Unet
from rdbm import RDBM
from train import Trainer

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_GENERATION_DIR = (SCRIPT_DIR / ".." / "data-generation").resolve()
DEFAULT_XML = DATA_GENERATION_DIR / "radiograph_input.xml"
DEFAULT_RADIOGRAPH = DATA_GENERATION_DIR / "radiograph.py"
DEFAULT_CKPT = SCRIPT_DIR / "save_folder" / "radio_4" / "model-10000" / "model-10000.pt"


def _set_object_material(root: ET.Element, compound: str, density: float) -> None:
    obj = root.find("Object")
    if obj is None:
        raise ValueError("No <Object> in XML template")
    mats = obj.find("Materials")
    if mats is not None:
        obj.remove(mats)
    mats = ET.SubElement(obj, "Materials")
    mat = ET.SubElement(mats, "Material")
    mat.set("compound", compound)
    mat.set("density", str(density))


def resolve_material(args) -> tuple[str, float] | None:
    """Return (compound, density) only when the user overrides the XML default."""
    if args.material is None and args.density is None:
        return None
    compound = args.material if args.material is not None else "SS316"
    if args.density is not None:
        density = args.density
    else:
        density = 19.3 if compound.strip().upper() == "W" else 7.93
    return compound, density


def make_config_xml(
    template_xml: Path,
    stl_path: Path,
    output_folder: Path,
    dest_xml: Path,
    material: tuple[str, float] | None = None,
) -> None:
    tree = ET.parse(template_xml)
    root = tree.getroot()

    stl_node = root.find("./Object/Geometry/STL")
    if stl_node is None:
        raise ValueError(f"No <Object>/<Geometry>/<STL> in {template_xml}")
    stl_node.set("root_path", str(stl_path.parent.resolve()))
    if stl_node.get("units") is None:
        stl_node.set("units", "mm")
    for child in list(stl_node.findall("File")):
        stl_node.remove(child)
    file_el = ET.SubElement(stl_node, "File")
    file_el.text = stl_path.name

    if material is not None:
        _set_object_material(root, material[0], material[1])

    out_node = root.find("Output")
    if out_node is None:
        raise ValueError(f"No <Output> in {template_xml}")
    out_node.set("folder", str(output_folder.resolve()))

    dest_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dest_xml, encoding="utf-8", xml_declaration=True)


XVFB_DISPLAY = ":99"
GLDISPATCH_SO = Path("/lib/x86_64-linux-gnu/libGLdispatch.so.0")
LOCAL_XVFB = Path.home() / ".local/xvfb/usr/bin/Xvfb"


def _xvfb_bin() -> Path | None:
    found = shutil.which("Xvfb")
    if found:
        return Path(found)
    return LOCAL_XVFB if LOCAL_XVFB.is_file() else None


def _x11_socket_exists(display: str) -> bool:
    num = display.split(".")[0].lstrip(":")
    return Path(f"/tmp/.X11-unix/X{num}").exists()


def ensure_radiograph_gl_env(env: dict) -> dict:
    """Give GVXR a usable GLFW/OpenGL context in this headless session.

    NVIDIA EGL on the DRM render nodes still fails to produce a live GL
    context here (GLEW/GLAD never bind). Xvfb + Mesa llvmpipe works if we
    also preload the system libGLdispatch so GVXR does not mix two copies.
    """
    env = dict(env)
    xvfb = _xvfb_bin()
    if xvfb is not None:
        env["PATH"] = str(xvfb.parent) + os.pathsep + env.get("PATH", "")
        if not _x11_socket_exists(XVFB_DISPLAY):
            log_path = Path("/tmp/eval_stl_xvfb.log")
            with open(log_path, "ab") as log:
                subprocess.Popen(
                    [
                        str(xvfb),
                        XVFB_DISPLAY,
                        "-screen",
                        "0",
                        "1024x768x24",
                        "+extension",
                        "GLX",
                        "-nolisten",
                        "tcp",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            for _ in range(30):
                if _x11_socket_exists(XVFB_DISPLAY):
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError(f"Xvfb failed to start on {XVFB_DISPLAY}; see {log_path}")
        env["DISPLAY"] = XVFB_DISPLAY
        env.setdefault("GVXR_RENDERER", "OPENGL")
        env.setdefault("__GLX_VENDOR_LIBRARY_NAME", "mesa")
    if GLDISPATCH_SO.is_file():
        prev = env.get("LD_PRELOAD", "")
        token = str(GLDISPATCH_SO)
        if token not in prev.split():
            env["LD_PRELOAD"] = token if not prev else f"{token} {prev}"
    return env


def run_radiograph(radiograph_py: Path, config_xml: Path, cwd: Path) -> None:
    env = ensure_radiograph_gl_env(os.environ.copy())
    cmd = [sys.executable, str(radiograph_py), str(config_xml)]
    print(f"Running: DISPLAY={env.get('DISPLAY')} {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"radiograph.py failed with code {proc.returncode}")


def find_pair(output_dir: Path, xml_stem: str) -> tuple[Path, Path]:
    """Locate noisy/GT PNGs written for this XML stem (proj0000 by default)."""
    noisy = sorted(output_dir.glob(f"{xml_stem}_proj*_noisy.png"))
    gt = sorted(output_dir.glob(f"{xml_stem}_proj*_ground_truth.png"))
    if not noisy or not gt:
        raise FileNotFoundError(
            f"Missing radiograph outputs under {output_dir} for stem={xml_stem}"
        )
    return noisy[0], gt[0]


def write_dataset_json(out_dir: Path, noisy: Path, gt: Path) -> Path:
    dataset_json = out_dir / "dataset.json"
    rel_in = noisy.name if noisy.parent == out_dir else str(noisy.relative_to(out_dir))
    rel_gt = gt.name if gt.parent == out_dir else str(gt.relative_to(out_dir))
    with open(dataset_json, "w", encoding="utf-8") as f:
        f.write(json.dumps({"input": rel_in, "target": rel_gt}) + "\n")
    return dataset_json


def build_trainer(dataset_json: Path, results_folder: Path, args) -> Trainer:
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=args.channels,
        condition=True,
        attn_heads=args.attn_heads,
        attn_dim_head=args.attn_dim_head,
    )
    diffusion = RDBM(
        model,
        image_size=args.crop_size,
        objective="pred_x_start",
        sampling_type="pred_x_start",
        timesteps=100,
        sampling_timesteps=10,
        condition=True,
    )
    use_amp = bool(args.amp) and args.mixed_precision != "no"
    mixed_precision = args.mixed_precision if use_amp else "no"
    trainer = Trainer(
        diffusion_model=diffusion,
        dataset_json=str(dataset_json),
        train_num_steps=1,
        train_batch_size=1,
        save_and_sample_every=1,
        results_folder=str(results_folder),
        optim="adam",
        train_lr=1e-4,
        amp=use_amp,
        mixed_precision_type=mixed_precision,
        crop_size=args.crop_size,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        split_ratio=0.5,
        split_seed=0,
        channels=args.channels,
        bit_depth=args.bit_depth,
        data_range=args.data_range,
        use_wandb=False,
    )
    return trainer


def load_checkpoint(trainer: Trainer, checkpoint: Path) -> int:
    """Load model/EMA weights from model-*.pt; skip optimizer (eval-only)."""
    name = checkpoint.stem  # model-10000
    if not name.startswith("model-"):
        raise ValueError(f"Expected checkpoint named model-<step>.pt, got {checkpoint.name}")
    step = int(name.split("-", 1)[1])
    data = torch.load(checkpoint, map_location=trainer.device, weights_only=False)
    trainer.model.load_state_dict(
        trainer._align_state_dict_to_module(data["model"], trainer.model)
    )
    trainer.ema.load_state_dict(
        trainer._align_state_dict_to_module(data["ema"], trainer.ema)
    )
    trainer.step = step
    return step


def run_compare(restored_dir: Path, data_root: Path, error_attenuation: bool = False) -> None:
    cmp = SCRIPT_DIR / "compare_restorations.py"
    cmd = [
        sys.executable,
        str(cmp),
        "--restored-dir",
        str(restored_dir),
        "--data-root",
        str(data_root),
        "--split",
        "eval",
    ]
    if error_attenuation:
        cmd.append("--error-attenuation")
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(SCRIPT_DIR), check=False)


def eval_one_stl(stl_path: Path, args) -> Path:
    stl_path = stl_path.resolve()
    if not stl_path.is_file():
        raise FileNotFoundError(stl_path)
    stem = stl_path.stem
    out_root = args.output_root.resolve()
    sim_dir = out_root / stem
    sim_dir.mkdir(parents=True, exist_ok=True)

    xml_stem = stem
    config_xml = sim_dir / f"{xml_stem}.xml"

    if not args.skip_sim:
        material = resolve_material(args)
        if material is not None:
            print(f"Material override: {material[0]} @ {material[1]} g/cm3")
        make_config_xml(args.xml.resolve(), stl_path, sim_dir, config_xml, material=material)
        run_radiograph(args.radiograph.resolve(), config_xml, cwd=DATA_GENERATION_DIR)
    else:
        print(f"[skip-sim] using existing radiographs under {sim_dir}")

    noisy, gt = find_pair(sim_dir, xml_stem)
    dataset_json = write_dataset_json(sim_dir, noisy, gt)
    print(f"dataset.json -> {dataset_json}")

    eval_results = out_root / "model_eval" / stem
    eval_results.mkdir(parents=True, exist_ok=True)

    set_seed(0)
    trainer = build_trainer(dataset_json, eval_results, args)
    step = load_checkpoint(trainer, args.checkpoint.resolve())
    print(f"Loaded checkpoint step={step}")

    # Write into eval_results/model-{step}/ as eval_*_restored.png
    # Single-pair dataset: mode='test' still returns the pair (fallback).
    trainer.results_folder = eval_results
    trainer.test(dataloader=trainer.dl_eval, split="eval")

    restored_dir = eval_results / f"model-{step}"
    if args.compare:
        run_compare(restored_dir, sim_dir, error_attenuation=args.error_attenuation)
    print(f"Done: {stl_path.name} -> {restored_dir}")
    return restored_dir


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stls", nargs="+", type=Path, help="One or more .stl paths")
    p.add_argument("--xml", type=Path, default=DEFAULT_XML, help="Radiograph template XML")
    p.add_argument("--radiograph", type=Path, default=DEFAULT_RADIOGRAPH, help="Path to radiograph.py")
    p.add_argument(
        "--output-root",
        type=Path,
        default=DATA_GENERATION_DIR / "eval_stl",
        help="Root for per-STL sim outputs and model eval",
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT, help="model-<step>.pt checkpoint")
    p.add_argument("--skip-sim", action="store_true", help="Skip radiograph.py; reuse existing PNGs")
    p.add_argument(
        "--material",
        type=str,
        default=None,
        help="Override object material compound (default: leave XML/SimConfig, SS316)",
    )
    p.add_argument(
        "--density",
        type=float,
        default=None,
        help="Override material density in g/cm3 (default: 19.3 if --material W, else 7.93)",
    )
    p.add_argument("--compare", action="store_true", default=True, help="Write comparison panels (default)")
    p.add_argument("--no-compare", action="store_false", dest="compare")
    p.add_argument(
        "--error-attenuation",
        action="store_true",
        help="Compare layout: 3x2 GT/rescaled/restored | errors/attenuation",
    )
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--attn_heads", type=int, default=4)
    p.add_argument("--attn_dim_head", type=int, default=16)
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--patch_stride", type=int, default=None)
    p.add_argument("--bit_depth", type=str, default="auto")
    p.add_argument("--data_range", type=float, default=None)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--mixed_precision", type=str, default="bf16")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        print(f"Error: checkpoint not found: {args.checkpoint}")
        return 1
    if not args.xml.is_file():
        print(f"Error: XML template not found: {args.xml}")
        return 1

    for stl in args.stls:
        print(f"\n======== {stl} ========")
        eval_one_stl(stl, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
