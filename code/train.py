import os
import re
import math
import time
import json
import warnings

import accelerate
import imageio.v3 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from ema_pytorch import EMA
from pathlib import Path
from skimage.metrics import structural_similarity
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler

from datasets_setting import RSDataset, peak_for_bit_depth, set_seed
from muon import Muon
from networks import Unet
from rdbm import RDBM

TARGET_ID_RE = re.compile(r"packed_calibration_target_(\d+)")

# Inductor may skip online-softmax when it splits reductions; harmless.
warnings.filterwarnings(
    "ignore",
    message=r"\s*Online softmax is disabled.*",
    category=UserWarning,
)

try:
    import wandb
except ImportError:
    wandb = None

parser = ArgumentParser()
parser.add_argument("--project_description", type=str, default="Residual Diffusion Bridge Model", help="Name of Project")
parser.add_argument("--exp_id", type=str, default="rdbm_radio_1", help="Experiment id / results subfolder")
parser.add_argument(
    "--dataset_json",
    type=str,
    help="Path to dataset.json (NDJSON input/target pairs)",
)
parser.add_argument("--channels", type=int, default=1, help="Image channels (1=grayscale, 3=RGB)")
parser.add_argument(
    "--bit_depth",
    type=str,
    default="auto",
    help="Input/output integer bit depth: auto|8|16. "
         "auto infers from dtype (uint8/uint16) or float magnitude; "
         "float/.npy ADU from radiograph.py typically uses peak 65535",
)
parser.add_argument(
    "--data_range",
    type=float,
    default=None,
    help="Override peak used to normalize inputs to [0,1] (e.g. 255, 65535). "
         "If omitted, inferred from --bit_depth / file dtype",
)
parser.add_argument("--attn_heads", type=int, default=4, help="Number of attention heads in Unet")
parser.add_argument("--attn_dim_head", type=int, default=32, help="Channels per attention head")
parser.add_argument("--crop_size", type=int, default=256, help="Training random crop size")
parser.add_argument("--patch_size", type=int, default=256, help="Eval tile size for stitched inference")
parser.add_argument(
    "--patch_stride",
    type=int,
    default=None,
    help="Eval tile stride (default: patch_size // 2 for overlap blending)",
)
parser.add_argument("--split_ratio", type=float, default=0.8, help="Train fraction of dataset.json pairs")
parser.add_argument("--split_seed", type=int, default=0, help="Seed for train/test split")
parser.add_argument(
    "--optim",
    type=str,
    default="muon",
    choices=["adam", "muon"],
    help="Optimizer: adam (original) or muon",
)
parser.add_argument("--lr", type=float, default=None, help="Learning rate (defaults: adam=8e-5, muon=1e-3)")
parser.add_argument("--wd", type=float, default=0.1, help="Weight decay (used by muon; ignored by adam)")
parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam / Muon-AdamW beta1")
parser.add_argument(
    "--adam_beta2",
    type=float,
    default=0.95,
    help="Adam / Muon-AdamW beta2",
)
parser.add_argument(
    "--lr_scheduler",
    type=str,
    default=None,
    help="LR scheduler (default: warmup_stable_decay for muon, none for adam). "
         "Examples: warmup_stable_decay, cosine, cosine_with_restarts, constant",
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=None,
    help="Warmup steps for the LR scheduler. "
         "If --warmup_ratio is set, that takes precedence.",
)
parser.add_argument(
    "--warmup_ratio",
    type=float,
    default=None,
    help="Warmup as a fraction of train_num_steps "
         "(default: 0.1 for muon warmup_stable_decay; overrides --warmup_steps)",
)
parser.add_argument(
    "--decay_ratio",
    type=float,
    default=None,
    help="Decay-phase fraction of train_num_steps for warmup_stable_decay "
         "(default: 0.5 for muon => stable until 50%%, then decay)",
)
parser.add_argument(
    "--min_lr_ratio",
    type=float,
    default=0.0,
    help="Final LR as a fraction of peak LR for warmup_stable_decay",
)
parser.add_argument("--train_num_steps", type=int, default=5001)
parser.add_argument("--train_batch_size", type=int, default=16)
parser.add_argument("--save_and_sample_every", type=int, default=1000, help="Eval + checkpoint interval (steps)")
parser.add_argument("--results_folder", type=str, default="./save_folder")
parser.add_argument("--amp", type=int, default=1, help="Enable mixed precision (1/0)")
parser.add_argument(
    "--mixed_precision",
    type=str,
    default="bf16",
    choices=["no", "fp16", "bf16"],
    help="Accelerate mixed precision dtype when --amp=1",
)
parser.add_argument("--compile", type=int, default=1, help="torch.compile the UNet (1/0)")
parser.add_argument(
    "--resume",
    type=str,
    default="auto",
    help="Resume checkpoint: 'auto' (latest model-*/model-*.pt), 'none', or a step number",
)
parser.add_argument("--use_wandb", type=int, default=1, help="Enable wandb logging (1/0)")
parser.add_argument("--wandb_project", type=str, default="radio", help="wandb project / workspace name")
parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity (optional)")
parser.add_argument("--wandb_name", type=str, default=None, help="wandb run name (default: exp_id)")
parser.add_argument("--wandb_id", type=str, default=None, help="wandb run id for resume (optional)")


def exists(x):
    return x is not None


def cycle(dl):
    while True:
        for data in dl:
            yield data


def divisible_by(numer, denom):
    return (numer % denom) == 0


def create_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)


def target_id_from_path(path):
    """Short id for eval outputs: numeric target, else STL/run stem."""
    m = TARGET_ID_RE.search(str(path))
    if m:
        return m.group(1)
    stem = Path(path).stem
    stem = re.sub(r"_proj\d+_ground_truth$", "", stem)
    stem = re.sub(r"_ground_truth$", "", stem)
    return stem


def find_latest_checkpoint_step(results_folder):
    """Return the largest step with model-{step}/model-{step}.pt, or None."""
    if not os.path.isdir(results_folder):
        return None
    best = None
    for name in os.listdir(results_folder):
        if not name.startswith("model-"):
            continue
        try:
            step = int(name.split("-", 1)[1])
        except ValueError:
            continue
        pt_path = os.path.join(results_folder, name, f"{name}.pt")
        if os.path.isfile(pt_path) and (best is None or step > best):
            best = step
    return best


def resolve_resume_step(results_folder, resume):
    """Parse --resume into a checkpoint step, or None to start fresh."""
    if resume is None:
        return None
    text = str(resume).strip().lower()
    if text in ("none", "no", "false", "0"):
        return None
    if text == "auto":
        return find_latest_checkpoint_step(results_folder)
    return int(resume)


def create_empty_json(json_path):
    with open(json_path, 'w') as file:
        pass


def write_json(json_path, item):
    with open(json_path, 'a+', encoding='utf-8') as f:
        line = json.dumps(item)
        f.write(line + '\n')


def readline_json(json_path, key=None):
    data = []
    with open(json_path, 'r') as f:
        items = f.readlines()
    if key is not None:
        for item in items:
            data.append(json.loads(item)[key])
        return np.asarray(data).mean()
    else:
        for item in items:
            data.append(json.loads(item))
        return data


def build_adam_optimizer(model, lr, adam_betas=(0.9, 0.99)):
    return Adam(model.parameters(), lr=lr, betas=adam_betas)


def build_muon_optimizer(model, lr, wd, adam_betas=(0.9, 0.99)):
    """Split params: Muon for >=2D, AdamW for the rest."""
    adam_keys = ["embed"]
    muon_params, adam_params = [], []
    muon_param_count = adam_param_count = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or any(s in name for s in adam_keys):
            adam_params.append(p)
            adam_param_count += p.numel()
        else:
            muon_params.append(p)
            muon_param_count += p.numel()

    print(f"Muon parameters: {muon_param_count:,}, Adam parameters: {adam_param_count:,}")
    return Muon(
        lr=lr,
        wd=wd,
        muon_params=muon_params,
        adamw_params=adam_params,
        adamw_betas=adam_betas,
        adamw_eps=1e-8,
    )


def build_optimizer(model, optim_type, lr, wd, adam_betas=(0.9, 0.99)):
    if optim_type == "adam":
        print(f"Using Adam optimizer (lr={lr}, betas={adam_betas})")
        return build_adam_optimizer(model, lr=lr, adam_betas=adam_betas)
    if optim_type == "muon":
        print(f"Using Muon optimizer (lr={lr}, wd={wd})")
        return build_muon_optimizer(model, lr=lr, wd=wd, adam_betas=adam_betas)
    raise ValueError(f"Unknown optimizer type: {optim_type}")


def resolve_scheduler_name(optim_type, lr_scheduler):
    """Default: warmup_stable_decay for muon, none for adam."""
    if lr_scheduler is not None and lr_scheduler != "" and lr_scheduler.lower() != "none":
        return lr_scheduler
    if optim_type == "muon":
        return "warmup_stable_decay"
    return None


def build_lr_scheduler(
    optimizer,
    scheduler_name,
    num_training_steps,
    warmup_steps=0,
    decay_steps=None,
    min_lr_ratio=0.0,
):
    if scheduler_name is None:
        return None
    scheduler_kwargs = {}
    if scheduler_name == "warmup_stable_decay":
        if decay_steps is None:
            raise ValueError("decay_steps is required for warmup_stable_decay")
        scheduler_kwargs = {
            "num_decay_steps": int(decay_steps),
            "min_lr_ratio": float(min_lr_ratio),
        }
        stable_steps = max(num_training_steps - warmup_steps - int(decay_steps), 0)
        print(
            f"Using LR scheduler '{scheduler_name}' "
            f"(warmup={warmup_steps}, stable={stable_steps}, decay={decay_steps}, "
            f"total={num_training_steps}, min_lr_ratio={min_lr_ratio})"
        )
    else:
        print(
            f"Using LR scheduler '{scheduler_name}' "
            f"(warmup_steps={warmup_steps}, num_training_steps={num_training_steps})"
        )
    return get_scheduler(
        name=scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        scheduler_specific_kwargs=scheduler_kwargs or None,
    )


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset_json,
        train_num_steps=100000,
        train_batch_size=1,
        save_and_sample_every=5000,
        results_folder='./results/',
        *,
        optim='muon',
        train_lr=1e-3,
        train_wd=0.1,
        adam_betas=(0.9, 0.95),
        lr_scheduler=None,
        warmup_steps=0,
        decay_steps=None,
        min_lr_ratio=0.0,
        ema_update_every=1,
        ema_decay=0.995,
        amp=True,
        mixed_precision_type='bf16',
        split_batches=True,
        max_grad_norm=1.,
        crop_size=256,
        patch_size=256,
        patch_stride=None,
        split_ratio=0.8,
        split_seed=0,
        channels=1,
        bit_depth="auto",
        data_range=None,
        use_wandb=False,
    ):
        super().__init__()

        self.accelerator = accelerate.Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no'
        )
        self.model = diffusion_model
        self.is_ddim_sampling = diffusion_model.is_ddim_sampling
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.max_grad_norm = max_grad_norm
        self.channels = channels
        self.bit_depth = bit_depth
        self.data_range = data_range
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.patch_stride = patch_size // 2 if patch_stride is None else patch_stride
        if self.patch_stride < 1 or self.patch_stride > self.patch_size:
            raise ValueError(
                f"patch_stride must be in [1, patch_size], got {self.patch_stride}"
            )
        # Only the main process logs to wandb.
        # Caller may pass True; we still gate on is_main_process here.
        self.use_wandb = bool(use_wandb) and self.accelerator.is_main_process

        self.dataset_json = dataset_json

        self.ds_train = RSDataset(
            dataset_json=self.dataset_json,
            mode='train',
            crop_size=crop_size,
            split_ratio=split_ratio,
            seed=split_seed,
            channels=channels,
            bit_depth=bit_depth,
            data_range=data_range,
        )
        self.dl_train = cycle(self.accelerator.prepare(
            DataLoader(self.ds_train, batch_size=train_batch_size, shuffle=True)
        ))
        # Shared peak for saving integer PNGs (from training split detection / config).
        self.output_peak = self.ds_train.detected_data_range
        if self.output_peak is None or self.output_peak <= 1.0 + 1e-6:
            # Already-normalized floats: default preview/export to 16-bit.
            self.output_peak = peak_for_bit_depth(16 if str(bit_depth) == "16" else 8)

        if self.accelerator.is_main_process:
            self.accelerator.print(
                'Training samples: {} (random {}x{} patches), data_range={:g}'.format(
                    len(self.ds_train), crop_size, crop_size, self.ds_train.detected_data_range
                )
            )

        # Held-out test split (full-res tiled PSNR/SSIM)
        self.ds_eval = RSDataset(
            dataset_json=self.dataset_json,
            mode='test',
            crop_size=crop_size,
            split_ratio=split_ratio,
            seed=split_seed,
            channels=channels,
            full_image=True,
            bit_depth=bit_depth,
            data_range=data_range if data_range is not None else self.ds_train.detected_data_range,
        )
        self.dl_eval = self.accelerator.prepare(DataLoader(self.ds_eval, batch_size=1))

        # Same metric protocol on the train split (no crop/aug) for comparison
        self.ds_eval_train = RSDataset(
            dataset_json=self.dataset_json,
            mode='train',
            crop_size=crop_size,
            split_ratio=split_ratio,
            seed=split_seed,
            channels=channels,
            full_image=True,
            bit_depth=bit_depth,
            data_range=data_range if data_range is not None else self.ds_train.detected_data_range,
        )
        self.dl_eval_train = self.accelerator.prepare(
            DataLoader(self.ds_eval_train, batch_size=1)
        )

        if self.accelerator.is_main_process:
            self.accelerator.print(
                'Eval splits (tiled {}x{}, stride {}): train={} test={}'.format(
                    self.patch_size,
                    self.patch_size,
                    self.patch_stride,
                    len(self.ds_eval_train),
                    len(self.ds_eval),
                )
            )

        self.opt = build_optimizer(
            diffusion_model,
            optim_type=optim,
            lr=train_lr,
            wd=train_wd,
            adam_betas=adam_betas,
        )
        scheduler_name = resolve_scheduler_name(optim, lr_scheduler)
        self.scheduler = build_lr_scheduler(
            self.opt,
            scheduler_name=scheduler_name,
            num_training_steps=train_num_steps,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            min_lr_ratio=min_lr_ratio,
        )
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
        self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.train_num_steps = train_num_steps
        self.step = 0

        # Do NOT accelerator.prepare() the LR scheduler. With multi-GPU and
        # split_batches=False (or mismatched prepare), AcceleratedScheduler
        # steps the underlying schedule num_processes times per call, so a
        # cosine meant for train_num_steps finishes ~N_gpu times too early.
        # We step the raw scheduler once per training iteration instead.
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        self._warmup_compiled_unet()

    def _unwrap_rdbm(self):
        """Unwrap DDP/DataParallel only.

        Do not use accelerator.unwrap_model here: with a nested torch.compile
        (UNet inside RDBM), accelerate's has_compiled_regions path looks for
        `_orig_mod` on the DDP parent and raises KeyError.
        """
        model = self.model
        if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
            return model.module
        return model

    def _unet_is_compiled(self):
        unet = getattr(self._unwrap_rdbm(), "model", None)
        if unet is None:
            return False
        # Whole-module or regional compile both introduce OptimizedModule (_orig_mod).
        return any(hasattr(m, "_orig_mod") for m in unet.modules())

    def _warmup_compiled_unet(self):
        """Run one autocast forward so the first compile specializes on bf16/fp16, not fp32."""
        if not self._unet_is_compiled():
            return
        device = self.device
        c = self.channels
        size = self.crop_size
        x = torch.randn(1, c, size, size, device=device)
        with torch.no_grad(), self.accelerator.autocast():
            # Run through the DDP-wrapped module (no accelerate unwrap).
            _ = self.model(img=[x, x])
        if self.accelerator.is_main_process:
            self.accelerator.print("torch.compile warmup forward done")

    @property
    def device(self):
        return self.accelerator.device

    @staticmethod
    def _strip_compile_prefix(state_dict):
        """Normalize torch.compile OptimizedModule keys (._orig_mod.) for portable checkpoints."""
        return {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}

    @staticmethod
    def _align_state_dict_to_module(state_dict, module):
        """Map portable keys (no _orig_mod) onto a live module that may be regionally compiled."""
        portable = Trainer._strip_compile_prefix(state_dict)
        live_by_portable = {
            k.replace("._orig_mod.", "."): k for k in module.state_dict().keys()
        }
        return {live_by_portable.get(k, k): v for k, v in portable.items()}

    def save(self, milestone=None):
        if not self.accelerator.is_local_main_process:
            return
        # Avoid accelerator.get_state_dict/unwrap_model: broken with nested compile.
        data = {
            'step': self.step,
            'model': self._strip_compile_prefix(self._unwrap_rdbm().state_dict()),
            'opt': self.opt.state_dict(),
            'ema': self._strip_compile_prefix(self.ema.state_dict()),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
        }
        checkpoint_save_path = os.path.join(self.results_folder, f'model-{milestone}')
        os.makedirs(checkpoint_save_path, exist_ok=True)
        torch.save(data, checkpoint_save_path + '/' + f'model-{milestone}.pt')

    def load(self, milestone=None):
        accelerator = self.accelerator
        device = accelerator.device
        checkpoint_save_path = os.path.join(self.results_folder, f'model-{milestone}')
        ckpt_path = os.path.join(checkpoint_save_path, f'model-{milestone}.pt')
        if accelerator.is_main_process:
            accelerator.print(f'Loading checkpoint: {ckpt_path}')
        data = torch.load(ckpt_path, map_location=device, weights_only=False)
        rdbm = self._unwrap_rdbm()
        rdbm.load_state_dict(self._align_state_dict_to_module(data['model'], rdbm))
        self.step = data['step'] + 1
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(self._align_state_dict_to_module(data['ema'], self.ema))
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
        if self.scheduler is not None and data.get('scheduler') is not None:
            try:
                self.scheduler.load_state_dict(data['scheduler'])
            except Exception as e:
                # e.g. cosine checkpoint -> warmup_stable_decay; re-sync by stepping.
                if accelerator.is_main_process:
                    accelerator.print(
                        f'Could not load scheduler state ({e}); '
                        f're-syncing schedule to step {self.step}'
                    )
                for _ in range(self.step):
                    self.scheduler.step()
        if accelerator.is_main_process:
            lr = self.opt.param_groups[0]["lr"]
            accelerator.print(f'Resumed at step {self.step}, lr={lr:.6e}')

    def cal_psnr(self, img_ref, img_gen, data_range=1.0):
        """PSNR on arrays already scaled so peak intensity equals data_range (default [0,1])."""
        ref = img_ref.astype(np.float32)
        gen = img_gen.astype(np.float32)
        mse = np.mean((ref - gen) ** 2) / (float(data_range) ** 2)
        if mse < 1.0e-12:
            return 100.0
        return 20.0 * math.log10(1.0 / math.sqrt(mse))

    def cal_ssim(self, img_ref, img_gen, data_range=1.0):
        if img_ref.ndim == 2:
            return structural_similarity(img_ref, img_gen, data_range=data_range)
        ssim_val = 0
        for i in range(img_ref.shape[-1]):
            ssim_val = ssim_val + structural_similarity(
                img_ref[:, :, i], img_gen[:, :, i], data_range=data_range
            )
        return ssim_val / img_ref.shape[-1]

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        track_metric_json_path = os.path.join(self.results_folder, 'metric.json')
        if self.accelerator.is_main_process:
            if not os.path.exists(track_metric_json_path):
                create_empty_json(track_metric_json_path)
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                self.model.train()
                file_input, file_target, file_name = next(self.dl_train)

                with self.accelerator.autocast():
                    self.opt.zero_grad()
                    loss = self.model(img=[file_target.to(device), file_input.to(device)])
                    self.accelerator.backward(loss)
                    accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.opt.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.ema.update()
                    loss_val = loss.item()
                    pbar.set_description(f'loss: {loss_val:.4f}')

                if self.use_wandb and wandb is not None and wandb.run is not None:
                    lr = self.opt.param_groups[0]["lr"]
                    wandb.log({
                        "train": {
                            "step": self.step,
                            "loss": loss_val,
                            "lr": lr,
                        }
                    }, step=self.step)

                if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                    self.test(dataloader=self.dl_eval_train, split='train')
                    self.test(dataloader=self.dl_eval, split='test')

                if self.accelerator.is_main_process:
                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        write_json(track_metric_json_path, f'model-{self.step} : ')
                        metrics_dir = os.path.join(self.results_folder, f'model-{self.step}')
                        train_psnr = float(readline_json(os.path.join(metrics_dir, 'train.json'), 'psnr'))
                        train_ssim = float(readline_json(os.path.join(metrics_dir, 'train.json'), 'ssim'))
                        test_psnr = float(readline_json(os.path.join(metrics_dir, 'test.json'), 'psnr'))
                        test_ssim = float(readline_json(os.path.join(metrics_dir, 'test.json'), 'ssim'))
                        accelerator.print(f'      train -> PSNR / SSIM -> {train_psnr:.6f} / {train_ssim:.6f}')
                        accelerator.print(f'      test  -> PSNR / SSIM -> {test_psnr:.6f} / {test_ssim:.6f}')
                        write_json(
                            track_metric_json_path,
                            f'    train -> PSNR / SSIM -> {train_psnr:.6f} / {train_ssim:.6f}',
                        )
                        write_json(
                            track_metric_json_path,
                            f'    test  -> PSNR / SSIM -> {test_psnr:.6f} / {test_ssim:.6f}',
                        )
                        if self.use_wandb and wandb is not None and wandb.run is not None:
                            wandb.log({
                                "metrics": {
                                    "step": self.step,
                                    "train_psnr": train_psnr,
                                    "train_ssim": train_ssim,
                                    "test_psnr": test_psnr,
                                    "test_ssim": test_ssim,
                                }
                            }, step=self.step)
                        accelerator.print('save model checkpoint')
                        self.save(self.step)

                self.step += 1
                pbar.update(1)

        accelerator.print('Training complete')
        if self.use_wandb and wandb is not None and wandb.run is not None:
            wandb.finish()

    def _patch_blend_weights(self, patch_size, device, dtype):
        """2D Hann weights for overlapping tile blend; ones if non-overlapping."""
        if self.patch_stride >= patch_size:
            return torch.ones(1, 1, patch_size, patch_size, device=device, dtype=dtype)
        wy = torch.hann_window(patch_size, periodic=False, device=device, dtype=dtype)
        wx = torch.hann_window(patch_size, periodic=False, device=device, dtype=dtype)
        w = wy[:, None] * wx[None, :]
        # Avoid exact zeros at tile edges so seams stay defined
        w = w.clamp_min(1e-3)
        return w.view(1, 1, patch_size, patch_size)

    @torch.no_grad()
    def sample_tiled(self, condi, patch_size=None, stride=None):
        """Run diffusion sampling on a grid of patches and stitch with blending.

        Args:
            condi: (1, C, H, W) degraded image in [0, 1]
        Returns:
            (1, C, H, W) restored image in [0, 1]
        """
        patch_size = self.patch_size if patch_size is None else patch_size
        stride = self.patch_stride if stride is None else stride
        assert condi.dim() == 4 and condi.shape[0] == 1, "expected batch size 1"

        _, c, h, w = condi.shape
        device = condi.device
        dtype = condi.dtype

        # Pad so the sliding window covers the full image
        pad_h = max(patch_size - h, 0)
        pad_w = max(patch_size - w, 0)
        if h > patch_size:
            rem = (h - patch_size) % stride
            if rem != 0:
                pad_h += stride - rem
        if w > patch_size:
            rem = (w - patch_size) % stride
            if rem != 0:
                pad_w += stride - rem

        condi_pad = F.pad(condi, (0, pad_w, 0, pad_h), mode='reflect')
        _, _, ph, pw = condi_pad.shape

        out = torch.zeros(1, c, ph, pw, device=device, dtype=dtype)
        weight = torch.zeros(1, 1, ph, pw, device=device, dtype=dtype)
        win = self._patch_blend_weights(patch_size, device, dtype)

        ys = list(range(0, ph - patch_size + 1, stride))
        xs = list(range(0, pw - patch_size + 1, stride))
        n_tiles = len(ys) * len(xs)
        tile_i = 0
        for y in ys:
            for x in xs:
                tile_i += 1
                patch = condi_pad[:, :, y:y + patch_size, x:x + patch_size]
                # Keep sampling under the same autocast dtype as training so
                # torch.compile does not re-specialize Float <-> BFloat16.
                with self.accelerator.autocast():
                    pred = self.ema.model.sample(patch)[1]
                if pred.dim() == 3:
                    pred = pred.unsqueeze(0)
                pred = pred.to(dtype=dtype)
                out[:, :, y:y + patch_size, x:x + patch_size] += pred * win
                weight[:, :, y:y + patch_size, x:x + patch_size] += win
                if tile_i == 1 or tile_i == n_tiles or tile_i % 16 == 0:
                    print(f"  tiled sample {tile_i}/{n_tiles}")

        out = out / weight.clamp_min(1e-8)
        return out[:, :, :h, :w]

    def test(self, dataloader, split):
        """Evaluate full-res tiled restores; write {split}_{id}_restored.png + {split}.json."""
        t0 = time.time() if self.accelerator.is_main_process else None
        out_dir = os.path.join(self.results_folder, f'model-{self.step}')
        create_folder(out_dir)
        metrics_path = os.path.join(out_dir, f'{split}.json')
        if self.accelerator.is_main_process:
            create_empty_json(metrics_path)
        self.ema.model.eval()
        for batch_id, (condi_tf, image_tf, name_path) in enumerate(dataloader):
            condi_tf = condi_tf.to(self.device)
            img_gen = self.sample_tiled(condi_tf)

            for i, path in enumerate(name_path):
                ref = self.tf2np(torch.clamp(image_tf[i:i + 1], 0., 1.))
                gen = self.tf2np(torch.clamp(img_gen[i:i + 1], 0., 1.))
                h, w = ref.shape[:2]
                gen = gen[:h, :w]

                psnr_val = self.cal_psnr(ref, gen, data_range=1.0)
                ssim_val = self.cal_ssim(ref, gen, data_range=1.0)

                tid = target_id_from_path(path)
                out_png = os.path.join(out_dir, f'{split}_{tid}_restored.png')
                try:
                    imageio.imwrite(out_png, self.float_to_int_image(gen))
                except Exception as e:
                    print(f'Warning: could not write {out_png}: {e}')

                write_json(metrics_path, {
                    'file_path': path,
                    'split': split,
                    'target_id': tid,
                    'psnr': psnr_val,
                    'ssim': ssim_val,
                    'bit_depth': self.bit_depth,
                    'data_range': self.output_peak,
                    'patch_size': self.patch_size,
                    'patch_stride': self.patch_stride,
                })
                print(f'[{split}] {batch_id} id={tid} PSNR/SSIM {psnr_val:.4f}/{ssim_val:.4f}')

        if self.accelerator.is_main_process:
            self.accelerator.print(f'{split} eval time : {time.time() - t0:.2f} s')

    def tf2np(self, image_tf):
        n, c, h, w = image_tf.size()
        assert n == 1
        if c == 1:
            image_np = image_tf.squeeze(0).squeeze(0).detach().cpu().numpy()
        else:
            image_np = image_tf.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        return image_np

    def float_to_int_image(self, image_np):
        """Quantize a float [0,1] image to uint8 or uint16 for PNG export."""
        peak = float(self.output_peak)
        if peak <= 255.0 + 1e-6:
            return np.clip(np.rint(image_np * 255.0), 0, 255).astype(np.uint8)
        return np.clip(np.rint(image_np * 65535.0), 0, 65535).astype(np.uint16)

    def tf2img(self, image_tf):
        image_np = self.tf2np(torch.clamp(image_tf, min=0., max=1.))
        return self.float_to_int_image(image_np)


def init_wandb(args, results_folder):
    """Minimal wandb init.

    Call only from the main process.
    """
    if not args.use_wandb:
        return False
    if wandb is None:
        print("Warning: --use_wandb=1 but wandb is not installed; skipping.")
        return False

    wandb_id_path = os.path.join(results_folder, "wandb_id.txt")
    if args.wandb_id:
        run_id = args.wandb_id
    elif os.path.isfile(wandb_id_path):
        with open(wandb_id_path, "r", encoding="utf-8") as f:
            run_id = f.read().strip()
    else:
        run_id = wandb.util.generate_id()
        with open(wandb_id_path, "w", encoding="utf-8") as f:
            f.write(run_id + "\n")
    run_name = args.wandb_name or args.exp_id
    init_kwargs = dict(
        project=args.wandb_project,
        resume="allow",
        id=run_id,
        name=run_name,
        notes=args.project_description,
        dir=str(results_folder),
        config={
            "exp_id": args.exp_id,
            "dataset_json": args.dataset_json,
            "channels": args.channels,
            "bit_depth": args.bit_depth,
            "data_range": args.data_range,
            "attn_heads": args.attn_heads,
            "attn_dim_head": args.attn_dim_head,
            "crop_size": args.crop_size,
            "patch_size": args.patch_size,
            "patch_stride": args.patch_stride,
            "optim": args.optim,
            "lr": args.lr,
            "wd": args.wd,
            "lr_scheduler": args.lr_scheduler,
            "warmup_steps": args.warmup_steps,
            "warmup_ratio": args.warmup_ratio,
            "decay_ratio": args.decay_ratio,
            "min_lr_ratio": args.min_lr_ratio,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "train_num_steps": args.train_num_steps,
            "train_batch_size": args.train_batch_size,
            "save_and_sample_every": args.save_and_sample_every,
            "amp": args.amp,
            "mixed_precision": args.mixed_precision,
            "compile": args.compile,
            "split_ratio": args.split_ratio,
            "split_seed": args.split_seed,
        },
    )
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity
    wandb.init(**init_kwargs)
    print(f"wandb: project={args.wandb_project} name={run_name} id={run_id}")
    return True


def train_ddp_accelerate(args):
    print('Procedure Running: ', args.project_description)
    print('Experiment: ', args.exp_id)
    print('Dataset JSON: ', args.dataset_json)
    print('Optimizer: ', args.optim)
    print('Train crop / eval patch: ', args.crop_size, args.patch_size, args.patch_stride)
    print('Bit depth / data_range: ', args.bit_depth, args.data_range)

    default_lrs = {'adam': 8e-5, 'muon': 1e-3}
    lr = args.lr if args.lr is not None else default_lrs[args.optim]
    args.lr = lr

    args.lr_scheduler = resolve_scheduler_name(args.optim, args.lr_scheduler)

    # Muon default: warmup 10% -> stable until 50% -> decay 50%.
    if args.warmup_ratio is not None:
        warmup_ratio = args.warmup_ratio
    elif args.warmup_steps is not None:
        warmup_ratio = None
    elif args.optim == "muon" and args.lr_scheduler == "warmup_stable_decay":
        warmup_ratio = 0.1
    else:
        warmup_ratio = 0.0

    if warmup_ratio is not None:
        warmup_steps = int(args.train_num_steps * warmup_ratio)
    else:
        warmup_steps = args.warmup_steps or 0

    if args.decay_ratio is not None:
        decay_ratio = args.decay_ratio
    elif args.optim == "muon" and args.lr_scheduler == "warmup_stable_decay":
        decay_ratio = 0.5
    else:
        decay_ratio = 0.0
    decay_steps = int(args.train_num_steps * decay_ratio)

    args.warmup_ratio = warmup_ratio
    args.warmup_steps = warmup_steps
    args.decay_ratio = decay_ratio
    stable_steps = max(args.train_num_steps - warmup_steps - decay_steps, 0)
    print(
        f"LR schedule: {args.lr_scheduler}, "
        f"warmup={warmup_steps} ({warmup_steps / max(args.train_num_steps, 1):.1%}), "
        f"stable={stable_steps} ({stable_steps / max(args.train_num_steps, 1):.1%}), "
        f"decay={decay_steps} ({decay_steps / max(args.train_num_steps, 1):.1%}), "
        f"beta2={args.adam_beta2}"
    )

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=args.channels,
        condition=True,
        attn_heads=args.attn_heads,
        attn_dim_head=args.attn_dim_head,
    )
    if args.compile:
        print("torch.compile: enabling on UNet (first steps will be slower while compiling)")
        model = torch.compile(model)
    diffusion = RDBM(
        model,
        image_size=args.crop_size,
        objective='pred_x_start',
        sampling_type='pred_x_start',
        timesteps=100,
        sampling_timesteps=10,
        condition=True,
    )
    results_folder = os.path.join(args.results_folder, args.exp_id)
    os.makedirs(results_folder, exist_ok=True)

    use_amp = bool(args.amp) and args.mixed_precision != "no"
    mixed_precision = args.mixed_precision if use_amp else "no"
    print(f"Mixed precision: {mixed_precision} (amp={int(use_amp)})")

    RDBM_Trainer = Trainer(
        diffusion_model=diffusion,
        dataset_json=args.dataset_json,
        train_num_steps=args.train_num_steps,
        train_batch_size=args.train_batch_size,
        save_and_sample_every=args.save_and_sample_every,
        results_folder=results_folder,
        optim=args.optim,
        train_lr=lr,
        train_wd=args.wd,
        adam_betas=(args.adam_beta1, args.adam_beta2),
        lr_scheduler=args.lr_scheduler,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        min_lr_ratio=args.min_lr_ratio,
        amp=use_amp,
        mixed_precision_type=mixed_precision,
        crop_size=args.crop_size,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
        channels=args.channels,
        bit_depth=args.bit_depth,
        data_range=args.data_range,
        use_wandb=bool(args.use_wandb),
    )
    if RDBM_Trainer.use_wandb:
        RDBM_Trainer.use_wandb = init_wandb(args, results_folder)

    resume_step = resolve_resume_step(results_folder, args.resume)
    if resume_step is not None:
        RDBM_Trainer.load(resume_step)
    elif RDBM_Trainer.accelerator.is_main_process:
        print(f"No checkpoint to resume (resume={args.resume!r}); starting from step 0")

    RDBM_Trainer.train()
    print('Procedure Termination: (Finished)')


if __name__ == '__main__':
    args = parser.parse_args()
    set_seed(0)
    train_ddp_accelerate(args)
