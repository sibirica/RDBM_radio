import os
import json
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import imageio.v3 as imageio


def exists(x):
    return x is not None


def cycle(dl):
    while True:
        for data in dl:
            yield data


def default(val, d):
    if exists(val) and (val is not None):
        return val
    return d() if callable(d) else d


def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def peak_for_bit_depth(bit_depth):
    """Linear peak value for an integer bit depth (8 -> 255, 16 -> 65535)."""
    bit_depth = int(bit_depth)
    if bit_depth not in (8, 16):
        raise ValueError(f"bit_depth must be 8 or 16, got {bit_depth}")
    return float((1 << bit_depth) - 1)


def infer_data_range(array, data_range=None, bit_depth="auto"):
    """Infer the divisor that maps image samples into [0, 1].

    Priority:
      1. explicit data_range
      2. bit_depth in {8, 16}
      3. dtype / magnitude heuristics (uint8/uint16/float ADU / already-normalized)
    """
    if data_range is not None:
        return float(data_range)
    if bit_depth not in (None, "auto", "Auto"):
        return peak_for_bit_depth(bit_depth)

    arr = np.asarray(array)
    if np.issubdtype(arr.dtype, np.integer):
        return float(np.iinfo(arr.dtype).max)

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    mx = float(finite.max())
    if mx <= 1.0 + 1e-3:
        return 1.0
    if mx <= 255.0 + 1e-3:
        return 255.0
    return 65535.0


def read_image_array(path):
    """Load PNG/NPY (and other imageio-readable) arrays without normalizing."""
    path = str(path)
    if path.lower().endswith(".npy"):
        return np.load(path)
    return np.asarray(imageio.imread(path))


class RSDataset(Dataset):
    """Paired NDJSON dataset: {"input": "...", "target": "..."}; supports 8/16-bit PNG and .npy."""

    def __init__(
        self,
        dataset_json,
        mode='train',
        patch_size=256,
        split_ratio=0.8,
        seed=0,
        channels=1,
        full_image=False,
        bit_depth="auto",
        data_range=None,
    ):
        super().__init__()
        assert mode in ['train', 'test']
        assert 0.0 < split_ratio < 1.0
        assert channels in (1, 3)

        self.dataset_json = os.path.abspath(dataset_json)
        self.root_dir = os.path.dirname(self.dataset_json)
        self.mode = mode
        self.full_image = full_image  # skip crop/aug; full-res for metrics
        self.patch_size = patch_size
        self.channels = channels
        self.bit_depth = bit_depth
        self.data_range = data_range
        self.transform = T.ToTensor()

        pairs = self.load_json(self.dataset_json)
        assert pairs, f"dataset.json is empty: {self.dataset_json}"

        rng = random.Random(seed)
        indices = list(range(len(pairs)))
        rng.shuffle(indices)
        n_train = max(1, int(round(len(pairs) * split_ratio)))
        if len(pairs) > 1:
            n_train = min(n_train, len(pairs) - 1)
        selected = set(indices[:n_train] if mode == 'train' else indices[n_train:])
        if mode == 'test' and not selected:
            selected = {indices[-1]}

        self.input_path = [os.path.join(self.root_dir, pairs[i]['input']) for i in sorted(selected)]
        self.target_path = [os.path.join(self.root_dir, pairs[i]['target']) for i in sorted(selected)]
        print(
            f"RSDataset mode={mode} pairs={len(self.input_path)} "
            f"(total={len(pairs)}, split_ratio={split_ratio}, seed={seed}, bit_depth={bit_depth})"
        )

        # Cache decoded+normalized arrays (shared data_range across the split).
        raw_pairs = []
        ranges = []
        for input_path, target_path in zip(self.input_path, self.target_path):
            raw_in = read_image_array(input_path)
            raw_gt = read_image_array(target_path)
            ranges.append(infer_data_range(raw_in, data_range=self.data_range, bit_depth=self.bit_depth))
            ranges.append(infer_data_range(raw_gt, data_range=self.data_range, bit_depth=self.bit_depth))
            raw_pairs.append((raw_in, raw_gt, input_path, target_path))

        self.detected_data_range = float(self.data_range) if self.data_range is not None else float(max(ranges))
        self._cache = []
        for raw_in, raw_gt, input_path, target_path in raw_pairs:
            im_degrade = self._normalize_image(raw_in, self.detected_data_range, input_path)
            im_clean = self._normalize_image(raw_gt, self.detected_data_range, target_path)
            if im_degrade.shape != im_clean.shape:
                raise ValueError(
                    f"Unmatched image sizes: {target_path} {im_clean.shape} vs "
                    f"{input_path} {im_degrade.shape}"
                )
            self._cache.append((im_degrade, im_clean))
        nbytes = sum(a.nbytes + b.nbytes for a, b in self._cache)
        print(
            f"RSDataset mode={mode}: cached {len(self._cache)} pairs "
            f"({nbytes / 1e6:.1f} MB), data_range={self.detected_data_range:g}"
        )

    def __len__(self):
        assert len(self.target_path) > 0, "selected split is empty"
        return len(self.target_path)

    def __getitem__(self, index):
        im_degrade, im_clean = self._cache[index]
        im_path = self.target_path[index]

        if self.mode == 'train' and not self.full_image:
            im_degrade, im_clean = self.random_crop_size(im_degrade, im_clean, self.patch_size)
            im_degrade, im_clean = self.random_dihedral_augment(im_degrade, im_clean)

        if self.transform is not None:
            im_degrade = self.transform(im_degrade)
            im_clean = self.transform(im_clean)
        return im_degrade, im_clean, im_path

    def _normalize_image(self, sample, data_range, path=""):
        """Raw array -> float32 HxWxC in [0, 1]."""
        sample = np.asarray(sample)
        if sample.ndim == 3:
            if sample.shape[2] >= 3 and self.channels == 1:
                rgb = sample[:, :, :3].astype(np.float32)
                sample = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
            elif sample.shape[2] >= 3 and self.channels == 3:
                sample = sample[:, :, :3]
            elif sample.shape[2] == 1:
                sample = sample[:, :, 0]
            else:
                raise ValueError(f"Unsupported image shape {sample.shape} for {path}")
        elif sample.ndim != 2:
            raise ValueError(f"Unsupported image ndim {sample.ndim} for {path}")

        sample = sample.astype(np.float32)
        if data_range <= 1.0 + 1e-6:
            sample = np.clip(sample, 0.0, 1.0)
        else:
            sample = np.clip(sample, 0.0, data_range) / data_range

        if self.channels == 1:
            if sample.ndim == 2:
                sample = sample[:, :, None]
            elif sample.shape[2] != 1:
                raise ValueError(f"Expected 1 channel, got shape {sample.shape} for {path}")
        else:
            if sample.ndim == 2:
                sample = np.stack([sample, sample, sample], axis=-1)
            elif sample.shape[2] != 3:
                raise ValueError(f"Expected 3 channels, got shape {sample.shape} for {path}")
        return sample

    def load_image(self, path, data_range=None):
        raw = read_image_array(path)
        if data_range is None:
            data_range = self.detected_data_range
            if data_range is None:
                data_range = infer_data_range(raw, data_range=self.data_range, bit_depth=self.bit_depth)
        return self._normalize_image(raw, data_range, path)

    def load_json(self, json_path):
        data = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        if 'input' not in obj or 'target' not in obj:
                            raise KeyError(f"line {line_num}: missing input/target")
                        data.append(obj)
                except json.JSONDecodeError:
                    print(f"{line_num} line in JSON is invalid")
        return data

    def resize_shape(self, image, short_side_length=256):
        oldh, oldw = image.shape[0], image.shape[1]
        if min(oldh, oldw) < short_side_length:
            scale = short_side_length * 1.0 / min(oldh, oldw)
            newh, neww = int(oldh * scale + 0.5), int(oldw * scale + 0.5)
            # cv2.resize drops the singleton channel dim for HxWx1
            image = cv2.resize(image, (neww, newh))
            if image.ndim == 2:
                image = image[:, :, None]
        return image

    def random_crop_size(self, imageA, imageB, size):
        imageA = self.resize_shape(imageA, size)
        imageB = self.resize_shape(imageB, size)
        if imageA.shape != imageB.shape:
            raise ValueError(f'Unmatched crop sizes {imageA.shape} vs {imageB.shape}')
        h, w = imageB.shape[:2]
        y0 = np.random.randint(0, h - size + 1)
        x0 = np.random.randint(0, w - size + 1)
        return (
            imageA[y0:y0 + size, x0:x0 + size, :],
            imageB[y0:y0 + size, x0:x0 + size, :],
        )

    def random_dihedral_augment(self, imageA, imageB):
        """Apply a random D4 symmetry (k*90° rotation and/or reflection) to both images."""
        k = np.random.randint(0, 4)
        flip = np.random.rand() < 0.5

        def _apply(img):
            out = np.rot90(img, k=k, axes=(0, 1))
            if flip:
                out = np.flip(out, axis=1)
            return np.ascontiguousarray(out)

        return _apply(imageA), _apply(imageB)

    def _padding(self, data, padding_shape=32):
        h, w = data.shape[-2], data.shape[-1]

        if h % padding_shape == 0:
            pad_h = 0
        else:
            pad_h = padding_shape - h % padding_shape
        if w % padding_shape == 0:
            pad_w = 0
        else:
            pad_w = padding_shape - w % padding_shape

        pad_top = 0
        pad_bottom = pad_h
        pad_left = 0
        pad_right = pad_w
        data = F.pad(data, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return data


if __name__ == '__main__':
    set_seed(11)
    root = os.path.join(os.path.dirname(__file__), '..', 'data-generation', 'fix_test_v2', 'dataset.json')
    for mode in ['train', 'test']:
        ds = RSDataset(dataset_json=root, mode=mode, channels=1)
        print(len(ds))
        dataloader = DataLoader(ds, batch_size=1, pin_memory=True, shuffle=True)
        for i, (lq, hq, paths) in enumerate(dataloader):
            print(f"{mode} {paths[0]}: hq shape={hq.shape}, lq shape={lq.shape}")
            if i >= 2:
                break
