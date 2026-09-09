# RDBM for radiography

Denoising of simulated radiographs with a Residual Diffusion Bridge Model (RDBM).

This repo adapts the RDBM image-restoration framework of Wang et al. ([paper](https://arxiv.org/abs/2510.23116), [upstream code](https://github.com/MiliLab/RDBM)) to grayscale radiographic data.

```
RDBM_radio/
  code/                 training, eval, visualization
  data-generation/      STL generation/radiograph simulation (adapted from David and Madeline's code) and dataset helpers
```

Typical pipeline: generate STLs (`target_testing_set.py` generates STLs) → simulate radiographs (`run_radiograph_batch.py`) → build `dataset.json` (`organize_radio.sh`) → train (`train.sh`) → evaluate network on test data and generate comparison plots (`compare_restorations.py`) and/or run `eval_stl.py` on a new STL.

---

## `code/` — train and evaluate

### `train.py`

Training and checkpointed evaluation for the radiograph denoiser.

It constructs the conditional UNet (`networks.py`) wrapped by the RDBM diffusion process (`rdbm.py`), loads paired noisy/ground-truth radiograph pairs from a NDJSON `dataset.json`, and then trains the network. Periodically computes evaluation metrics (PSNR/SSIM) using tiled full-resolution inference on train and test splits, writes restored PNGs as `{train|test}_{id}_restored.png`, and saves checkpoint `model-<step>.pt` under `save_folder/<exp_id>/`.

Eval does not run the UNet on the whole image at once. `sample_tiled` covers the radiograph with a sliding window of `--patch_size` with step `--patch_stride`, denoises each tile, and stitches the results. When tiles overlap (`stride < patch_size`), each tile is weighted by a 2D Hann window (outer product of 1D Hann windows, floored so edges are not exact zeros). `--patch_stride` is the offset between adjacent tile centers; it defaults to `patch_size // 2` (50% overlap). A smaller stride means more overlap and fewer seams, at the cost of more tiles. If stride equals `patch_size` there is no overlap and no Hann weighting.

You can call it directly with many CLI flags (`--dataset_json`, `--optim`, `--bit_depth`, `--resume`, ...), but it is easiest to run via `train.sh`. Weights and Biases logging is implemented; pass `--use_wandb 1` to turn it off.

### `train.sh`

Shell launcher for `train.py` with the defaults used for the current training script (experiment name `radio_4`, Muon optimizer, bf16 automatic mixed precision, `torch.compile`, tile size 256², 10k training steps). Set `use_wandb=1` to enable wandb.

Run it from `code/`. It picks the GPU count automatically and starts `torchrun`. Edit this script, or overwrite with environment variables before invoking, for example:

```bash
expid=radio_5 dataset_json=../data-generation/fix_test_realistic_v2/dataset.json ./train.sh
```

### `rdbm.py`

Implements the residual diffusion bridge model from the Wang et al. paper.

### `networks.py`

Defines the conditional UNet that predicts the bridge residual / denoising target. The network has been modified to work with greyscale data (`channels=1`).

Some hyperparameters that can be edited: number of attention heads `--attn_heads` / dimension per head `--attn_dim_head`; `train.sh` defaults 4 and 16; UNet base width `dim` (currently 64) and per-layer scales `dim_mults` (currently `(1, 2, 4, 8)`). A longer `dim_mults` tuple adds downsampling stages (and the last multiplier sets how wide the bottleneck is).

### `datasets_setting.py`

Dataset I/O for training and eval. `RSDataset` reads a `dataset.json` (one JSON object per line with `"input"` / `"target"` paths relative to that file’s directory) and loads uint8 or uint16 PNGs or float `.npy` arrays.

The network expects intensities in `[0, 1]`, so every image is divided by a peak (white level). You can set that peak manually with `--data_range`. If you do not, `--bit_depth 8` or `16` uses 255 or 65535. If `--bit_depth` is `auto` (the default), the peak is guessed from the file: uint8 → 255, uint16 → 65535, floats already in `[0, 1]` are left alone, and other floats are treated as 16-bit ADU (divide by 65535). The same peak is used for every image in the run.

Then the dataset either samples random crops (controllable by `set_seed`) for training, or returns the full image for tiled evaluation.

### `muon.py`

Muon optimizer (used in `train.py`, do not change).

### `eval_stl.py`

End-to-end test on new geometry.

Given one or more `.stl` paths it:

1. Writes a temporary XML from `radiograph_input.xml` pointing at that STL  
2. Runs `data-generation/radiograph.py` to produce noisy + ground-truth PNGs  
3. Builds `dataset.json` for each STL
4. Loads a trained checkpoint (default: `save_folder/radio_4/model-10000`)  
5. Computes restored radiograph, writing `eval_{id}_restored.png`  
6. Optionally calls `compare_restorations.py` for visualization plots  

Use `--skip-sim` if the radiographs already exist. Outputs saved under `data-generation/eval_stl/` by default.

### `compare_restorations.py`

Visualization of already-generated restorations. Finds files named `{train|test|eval}_{id}_restored.png`, looks up matching ground truth (and, for the baseline, the noisy input) under `--data-root`, and writes comparison / error images. Does not run the model; it only plots results from `train.py` or `eval_stl.py`. Command examples for each layout are in the `compare_restorations.py` module docstring (`python compare_restorations.py -h`).

- **2x2** (default): ground truth, corrupted, restored, and `|GT − restored|`.
- **3x2** (`--error-attenuation`): adds the rescale-only baseline and an error-attenuation map.
- **1x2** (`--error-attenuation --layout 1x2`): ground truth and error attenuation only.

The rescale-only baseline is not a second model: it is the noisy radiograph after an affine intensity remap so the two dominant gray-level peaks line up with the two peaks of the ground truth. Peaks are taken from a KDE of the intensity histogram (not min/max), which typically corresponds to the open-beam / high-transmission mode and the densest parts of the object.

Error attenuation at each pixel is defined as `|ground_truth − restored| / max(|ground_truth − rescaled baseline|, epsilon)`.

**Example (run from `code/`):**

```bash
./train.sh

python compare_restorations.py \
  --restored-dir save_folder/radio_4/model-10000 \
  --data-root ../data-generation/fix_test_realistic_v2

python eval_stl.py ../data-generation/conebeam_abel_stepwedge_fixed.stl
```

---

## `data-generation/` — simulate radiographs

There have been minor updates to these scripts compared to the version David wrote.

### `radiograph.py`

Run with `python radiograph.py radiograph_input.xml`. Headless machines need a working EGL/OpenGL context (e.g. membership in the `render` group for `/dev/dri/renderD*`).

### `radiograph_input.xml`

Default simulation template: beam type and energy, object pose / jitter, STL path, detector size and bit depth, noise and blur knobs, output options. `run_radiograph_batch.py` and `eval_stl.py` clone and edit this rather than hand-writing a new XML each time.

### `run_radiograph_batch.py`

NEW: Batch runner for a directory of `.stl` files. For each mesh, it copies the template XML, sets `<STL>` to that file, sets a per-STL output folder under `--output-root`, and runs `radiograph.py`. Supports `--start-from` and `--keep-configs` for resuming long jobs.

### `organize_radio.sh`

Turns a folder of noisy/ground-truth pairs into a JSON specifying the training dataset. Edit `DATA_DIR` and `FORMAT` at the top of the script, or pass a folder as the first argument. It finds `*_noisy.{png|npy}` files up to one child folder deep with a matching `*_ground_truth` sibling, and writes NDJSON lines `{"input": "...", "target": "..."}` into that directory’s `dataset.json`. Nested trees deeper than one child folder are skipped — run the script again on those subtrees if needed.

```bash
./organize_radio.sh
./organize_radio.sh /path/to/data
FORMAT=npy ./organize_radio.sh /path/to/data
```

### `target_testing_set.py`

Procedural STL generator (unchanged from Madeline's version). Exports `.stl` files into an output directory (default `realistic_dataset/`). These STLs are fed to `run_radiograph_batch.py` / `radiograph.py`.

**Example simulation pipeline:**

```bash
cd data-generation
python target_testing_set.py
python run_radiograph_batch.py /path/to/stls --output-root fix_test_realistic_v2
./organize_radio.sh fix_test_realistic_v2
```

---

## Future work

The current network restores a single-channel radiograph (intensity). Future plans include **density estimation**: recover a per-pixel density from the restored intensities.

Two ways to attach that:

- Widen the UNet from one output channel to two, so it jointly predicts intensity and density. This needs paired density maps in `dataset.json` (or a second target array) and a loss on both heads (can adjust the ratios). The current `--channels` flag is only 1 vs 3 (RGB) and would need to allow a density channel in the RSDataset definition.
- Keep the existing 1-channel restorer and add a small convolutional module after the UNet that maps pixel values to densities. That head can be trained on its own (or jointly) without changing the diffusion backbone.

For **real experimental data**, data volume may be too small to train a full model. In that case, one should fine-tune from a checkpoint trained on simulation (`--resume`) rather than training from scratch. Consider freezing weights in some of the UNet layers.
