#!/usr/bin/env python3
"""
Run radiograph.py once per .stl in a directory.

For each STL, clones the template XML (default: radiograph_input.xml),
replaces the <STL>/<File> entry, sets root_path to the STL directory,
and writes outputs under <output_root>/<stl_stem>/.

Usage (from RDBM_radio/data-generation/):
  python run_radiograph_batch.py /path/to/stl_dir
  python run_radiograph_batch.py /path/to/stl_dir --xml radiograph_input.xml --output-root radiographs_batch
  python run_radiograph_batch.py /path/to/stl_dir --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_XML = SCRIPT_DIR / "radiograph_input.xml"
DEFAULT_RADIOGRAPH = SCRIPT_DIR / "radiograph.py"


def _natural_key(path: Path):
    """Sort key so calibration_target_9.stl comes before calibration_target_10.stl."""
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(p) if p.isdigit() else p for p in parts]


def list_stls(stl_dir: Path) -> list[Path]:
    files = list(stl_dir.glob("*.stl")) + list(stl_dir.glob("*.STL"))
    # de-dupe (case-insensitive FS), then natural-sort by filename
    seen = set()
    out = []
    for f in files:
        key = str(f.resolve()).lower()
        if key not in seen:
            seen.add(key)
            out.append(f)
    out.sort(key=_natural_key)
    return out


def make_config_xml(
    template_xml: Path,
    stl_path: Path,
    output_folder: Path,
    dest_xml: Path,
) -> None:
    tree = ET.parse(template_xml)
    root = tree.getroot()

    stl_node = root.find("./Object/Geometry/STL")
    if stl_node is None:
        raise ValueError(f"No <Object>/<Geometry>/<STL> node in {template_xml}")

    stl_node.set("root_path", str(stl_path.parent.resolve()))
    # Keep existing units attribute if present
    if stl_node.get("units") is None:
        stl_node.set("units", "mm")

    # Replace all <File> children with this STL basename
    for child in list(stl_node.findall("File")):
        stl_node.remove(child)
    file_el = ET.SubElement(stl_node, "File")
    file_el.text = stl_path.name

    out_node = root.find("Output")
    if out_node is None:
        raise ValueError(f"No <Output> node in {template_xml}")
    out_node.set("folder", str(output_folder.resolve()))

    dest_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dest_xml, encoding="utf-8", xml_declaration=True)


def run_one(
    radiograph_py: Path,
    config_xml: Path,
    cwd: Path,
    dry_run: bool,
) -> int:
    cmd = [sys.executable, str(radiograph_py), str(config_xml)]
    print(f"\n=== {'DRY-RUN ' if dry_run else ''}Running: {' '.join(cmd)} ===")
    if dry_run:
        return 0
    env = os.environ.copy()
    # Prefer headless OpenGL/EGL when no display is available
    if not env.get("DISPLAY") and not env.get("WAYLAND_DISPLAY"):
        env.setdefault("GVXR_FORCE_WINDOW", "0")
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-run radiograph.py over every .stl in a directory."
    )
    parser.add_argument(
        "stl_dir",
        type=Path,
        help="Directory containing .stl files",
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=DEFAULT_XML,
        help=f"Template XML config (default: {DEFAULT_XML})",
    )
    parser.add_argument(
        "--radiograph",
        type=Path,
        default=DEFAULT_RADIOGRAPH,
        help=f"Path to radiograph.py (default: {DEFAULT_RADIOGRAPH})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Root directory for per-STL output folders "
            "(default: <template Output folder>_batch next to template, "
            "or ./radiographs_batch)"
        ),
    )
    parser.add_argument(
        "--keep-configs",
        action="store_true",
        help="Keep generated per-STL XML configs under <output-root>/_configs/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without running radiograph.py",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        help="Skip STLs until this basename is reached (inclusive)",
    )
    args = parser.parse_args()

    stl_dir = args.stl_dir.resolve()
    if not stl_dir.is_dir():
        print(f"Error: stl_dir is not a directory: {stl_dir}", file=sys.stderr)
        return 1
    if not args.xml.is_file():
        print(f"Error: template XML not found: {args.xml}", file=sys.stderr)
        return 1
    if not args.radiograph.is_file():
        print(f"Error: radiograph.py not found: {args.radiograph}", file=sys.stderr)
        return 1

    stls = list_stls(stl_dir)
    if not stls:
        print(f"Error: no .stl files found in {stl_dir}", file=sys.stderr)
        return 1

    if args.output_root is not None:
        output_root = args.output_root.resolve()
    else:
        # Prefer extending the template's Output/@folder if present
        try:
            tmpl = ET.parse(args.xml).getroot()
            out = tmpl.find("Output")
            base = out.get("folder") if out is not None else None
        except Exception:
            base = None
        if base:
            output_root = (SCRIPT_DIR / f"{base}_batch").resolve()
        else:
            output_root = (SCRIPT_DIR / "radiographs_batch").resolve()

    output_root.mkdir(parents=True, exist_ok=True)
    config_dir = output_root / "_configs"
    if args.keep_configs:
        config_dir.mkdir(parents=True, exist_ok=True)

    if args.start_from:
        start_name = args.start_from
        filtered = []
        started = False
        for p in stls:
            if not started and p.name == start_name:
                started = True
            if started:
                filtered.append(p)
        if not started:
            print(f"Error: --start-from '{start_name}' not found in {stl_dir}", file=sys.stderr)
            return 1
        stls = filtered

    print(f"STL directory : {stl_dir}")
    print(f"Template XML  : {args.xml.resolve()}")
    print(f"radiograph.py : {args.radiograph.resolve()}")
    print(f"Output root   : {output_root}")
    print(f"Found {len(stls)} STL file(s)")

    failures: list[tuple[str, int]] = []
    tmp_root = None
    try:
        if not args.keep_configs:
            tmp_root = tempfile.mkdtemp(prefix="radiograph_batch_")
            config_parent = Path(tmp_root)
        else:
            config_parent = config_dir

        for i, stl_path in enumerate(stls, start=1):
            stem = stl_path.stem
            out_folder = output_root / stem
            config_xml = config_parent / f"{stem}.xml"

            print(f"\n[{i}/{len(stls)}] {stl_path.name}")
            print(f"  output -> {out_folder}")
            print(f"  config -> {config_xml}")

            make_config_xml(args.xml, stl_path, out_folder, config_xml)
            rc = run_one(args.radiograph, config_xml, SCRIPT_DIR, args.dry_run)
            if rc != 0:
                print(f"  FAILED (exit {rc}): {stl_path.name}", file=sys.stderr)
                failures.append((stl_path.name, rc))
            else:
                print(f"  OK: {stl_path.name}")
    finally:
        if tmp_root is not None and not args.keep_configs:
            shutil.rmtree(tmp_root, ignore_errors=True)

    print("\n========== Batch summary ==========")
    print(f"Succeeded: {len(stls) - len(failures)} / {len(stls)}")
    if failures:
        print("Failures:")
        for name, rc in failures:
            print(f"  {name} (exit {rc})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
