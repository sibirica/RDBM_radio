# RDBM for radiography

Denoising of simulated radiographs with a Residual Diffusion Bridge Model (RDBM).

This repo adapts the RDBM image-restoration framework of Wang et al. ([paper](https://arxiv.org/abs/2510.23116), [upstream code](https://github.com/MiliLab/RDBM)) to grayscale radiographic data: synthetic noisy–clean pairs from GVXR, training, tiled full-image inference, and comparison figures.

```
RDBM_radio/
  code/                 training, eval, visualization
  data-generation/      STL → radiograph simulation and dataset helpers
```

Workflow: generate STLs → simulate radiographs → build `dataset.json` → train → evaluate network on test data (or run `eval_stl.py` on a held-out STL).

---

## `code/` — train and evaluate

### `train.py`

Training and checkpointed evaluation for the radiography denoiser.

It constructs the conditional UNet (`networks.py`) wrapped by the RDBM diffusion process (`rdbm.py`), loads paired noisy/GT images via `RSDataset` from a NDJSON `dataset.json`, and optimizes with Adam or Muon. During training it periodically runs **tiled** full-resolution inference on train and test splits (overlapping patches blended together), writes restored PNGs as `{train|test}_{id}_restored.png`, logs PSNR/SSIM, and saves `model-<step>.pt` under `save_folder/<exp_id>/`.

You can call it directly with many CLI flags (`--dataset_json`, `--optim`, `--bit_depth`, `--resume`, …), but day-to-day use is usually through `train.sh`.

### `train.sh`

Shell launcher for `train.py` with the defaults used for the current radiography experiments (e.g. experiment id `radio_4`, Muon, bf16 AMP, `torch.compile`, wandb project `radio`, 256² crops/tiles, ~10k steps).

Run it from `code/`. It picks the GPU count automatically and starts `torchrun`. Override any setting with environment variables before invoking, for example:

```bash
expid=radio_5 dataset_json=../data-generation/fix_test_realistic_v2/dataset.json ./train.sh
```

### `rdbm.py`

Implements the residual diffusion bridge: how noise is injected and removed as a function of the residual between the degraded observation and the clean target, plus the sampling loop used at inference time. This is the algorithmic core from the Wang et al. paper; `train.py` treats it as the diffusion model around the UNet.

### `networks.py`

Defines the conditional UNet that predicts the bridge residual / denoising target. Conditioning is the degraded radiograph; the network is grayscale-capable (`channels=1` for this project).

### `datasets_setting.py`

Dataset I/O for training and eval. `RSDataset` reads a `dataset.json` (one JSON object per line with `"input"` / `"target"` paths relative to that file’s directory), loads uint8 or uint16 PNGs or float `.npy` arrays, normalizes them with a shared `data_range` / bit-depth convention, and either:

- samples random crops for training, or
- returns the full image for tiled evaluation.

Also provides small helpers (`set_seed`, peak inference for bit depth).

### `muon.py`

Muon optimizer (orthogonalized updates via Newton–Schulz), used when `--optim muon` / `optim=muon` in `train.sh`. Hybrid with AdamW for embeddings and low-rank params is wired in `train.py`.

### `eval_stl.py`

End-to-end test on geometries that were **not** necessarily in the training set.

Given one or more `.stl` paths it:

1. Writes a temporary XML from `radiograph_input.xml` pointing at that STL  
2. Runs `data-generation/radiograph.py` to produce noisy + ground-truth PNGs  
3. Builds a one-pair `dataset.json`  
4. Loads a trained checkpoint (default: `save_folder/radio_4/model-10000`)  
5. Runs the same tiled restore path as test eval, writing `eval_<stem>_restored.png`  
6. Optionally calls `compare_restorations.py` for panels  

Use `--skip-sim` if the radiographs already exist. Outputs land under `data-generation/eval_stl/` by default.

### `compare_restorations.py`

Offline visualization of already-written restores. It finds files named `{train|test|eval}_{id}_restored.png`, looks up matching ground truth under `--data-root`, and writes comparison / error images (absolute error normalized by the GT intensity range, typically with a log-scaled colorbar). Optional `--error-attenuation` layouts compare “rescale-only” baselines to the network restore.

Does not run the model; it only plots results after `train.py` or `eval_stl.py`.

**Examples (from `code/`):**

```bash
./train.sh

python compare_restorations.py \
  --restored-dir save_folder/radio_4/model-10000 \
  --data-root ../data-generation/fix_test_realistic_v2

python eval_stl.py ../data-generation/conebeam_abel_stepwedge_fixed.stl
```

---

## `data-generation/` — simulate radiographs

Scripts and configs live at the **root** of this folder. Large generated datasets (`fix_test_*`, `eval_stl/`, …) are meant to stay local and are gitignored.

### `radiograph.py`

Single-run radiograph simulator driven by an XML config. Uses [gVirtualXRay](https://gvirtualxray.sourceforge.io/) (GVXR) to project an STL under the configured source / detector / material settings, then can add blur, focal-spot effects, and detector noise. Writes paired `*_ground_truth` and `*_noisy` images (16-bit PNG and/or `.npy`) plus a small config JSON sidecar into the folder named in the XML `<Output>` element.

Typical invocation: `python radiograph.py radiograph_input.xml` (or a per-STL XML produced by the batch runner). Headless machines need a working EGL/OpenGL context (e.g. membership in the `render` group for `/dev/dri/renderD*`).

### `radiograph_input.xml`

Default simulation template: beam type and energy, object pose / jitter, STL path, detector size and bit depth, noise and blur knobs, output options. `run_radiograph_batch.py` and `eval_stl.py` clone and edit this rather than hand-writing a new XML each time.

### `run_radiograph_batch.py`

Batch driver over a directory of `.stl` files. For each mesh it copies the template XML, sets `<STL>` to that file, sets a per-STL output folder under `--output-root`, and invokes `radiograph.py`. Supports `--dry-run`, `--start-from`, and `--keep-configs` for long jobs.

### `organize_radio.sh`

Turns a folder of simulated pairs into a training manifest. It finds `*_noisy.{png|npy}` files up to depth 2 under the given directory, requires a matching `*_ground_truth` sibling, and appends NDJSON lines `{"input": "...", "target": "..."}` into that directory’s `dataset.json`. Nested trees deeper than one child folder are skipped on purpose—run the script again on those subtrees if needed.

```bash
./organize_radio.sh fix_test_realistic_v2
FORMAT=npy ./organize_radio.sh /path/to/data
```

### `target_testing_set.py`

Procedural phantom generator: builds varied solid meshes with GMSH (packed targets, step-like structures, etc.) and exports `.stl` files into an output directory (default `realistic_dataset/`). Those STLs are then fed to `run_radiograph_batch.py` / `radiograph.py`.

**Example simulation pipeline:**

```bash
cd data-generation
python target_testing_set.py          # optional: new STLs
python run_radiograph_batch.py /path/to/stls --output-root fix_test_realistic_v2
./organize_radio.sh fix_test_realistic_v2
```

---

## Citation

RDBM method and original implementation:

```bibtex
@inproceedings{wang2026residual,
  title={Residual diffusion bridge model for image restoration},
  author={Wang, Hebaixu and Zhang, Jing and Chen, Haoyang and Guo, Haonan and Wang, Di and Ma, Jiayi and Du, Bo},
  booktitle={Proceedings of the Conference on Computer Vision and Pattern Recognition},
  pages={8375--8386},
  year={2026}
}
```
