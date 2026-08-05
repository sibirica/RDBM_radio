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


class RSDataset(Dataset):
    """Paired image dataset driven by a single dataset.json (NDJSON).

    Each line: {"input": "in/foo.png", "target": "target/foo.png"}
    Paths are resolved relative to the directory containing dataset.json.
    Train/test split is computed automatically (seeded shuffle + ratio).
    """

    def __init__(
        self,
        dataset_json,
        mode='train',
        crop_size=256,
        split_ratio=0.8,
        seed=0,
        channels=1,
        full_image=False,
    ):
        super().__init__()
        assert mode in ['train', 'test']
        assert 0.0 < split_ratio < 1.0
        assert channels in (1, 3)

        self.dataset_json = os.path.abspath(dataset_json)
        self.root_dir = os.path.dirname(self.dataset_json)
        self.mode = mode
        # full_image=True: return full-res (for PSNR/SSIM); skip train crop/aug
        self.full_image = full_image
        self.crop_size = crop_size
        self.channels = channels
        self.transform = T.ToTensor()

        pairs = self.load_json(self.dataset_json)
        assert len(pairs) > 0, f"dataset.json is empty: {self.dataset_json}"

        rng = random.Random(seed)
        indices = list(range(len(pairs)))
        rng.shuffle(indices)

        n_train = max(1, int(round(len(pairs) * split_ratio)))
        if len(pairs) > 1:
            n_train = min(n_train, len(pairs) - 1)
        train_idx = set(indices[:n_train])
        selected = train_idx if mode == 'train' else set(indices[n_train:])
        if mode == 'test' and len(selected) == 0:
            selected = {indices[-1]}

        self.input_path = []
        self.target_path = []
        for i in sorted(selected):
            item = pairs[i]
            self.input_path.append(os.path.join(self.root_dir, item['input']))
            self.target_path.append(os.path.join(self.root_dir, item['target']))

        assert len(self.input_path) == len(self.target_path)
        print(
            f"RSDataset mode={mode} pairs={len(self.input_path)} "
            f"(total={len(pairs)}, split_ratio={split_ratio}, seed={seed}, "
            f"full_image={full_image})"
        )

        # Decode once into RAM; PNG I/O dominated step time on 2k radiographs.
        self._cache = []
        for input_path, target_path in zip(self.input_path, self.target_path):
            im_degrade = self.load_image(input_path)
            im_clean = self.load_image(target_path)
            if not self.check_size(im_degrade, im_clean):
                raise ValueError(
                    f"Unmatched image sizes: {target_path} {im_clean.shape} vs "
                    f"{input_path} {im_degrade.shape}"
                )
            self._cache.append((im_degrade, im_clean))
        nbytes = sum(a.nbytes + b.nbytes for a, b in self._cache)
        print(f"RSDataset mode={mode}: cached {len(self._cache)} pairs ({nbytes / 1e6:.1f} MB)")

    def __len__(self):
        assert len(self.target_path) > 0, "selected split is empty"
        return len(self.target_path)

    def __getitem__(self, index):
        im_degrade, im_clean = self._cache[index]
        im_path = self.target_path[index]

        if self.mode == 'train' and not self.full_image:
            # Random patch + D4 augmentation for training only.
            # full_image=True keeps the train-split images full-res for metrics.
            im_degrade, im_clean = self.random_crop_size(im_degrade, im_clean, self.crop_size)
            im_degrade, im_clean = self.random_dihedral_augment(im_degrade, im_clean)

        if self.transform is not None:
            im_degrade = self.transform(im_degrade)
            im_clean = self.transform(im_clean)

        return im_degrade, im_clean, im_path

    def check_size(self, input, target):
        if input.shape != target.shape:
            return 0
        return 1

    def load_image(self, path, data_range=255.0):
        sample = imageio.imread(path)
        sample = np.asarray(sample)

        if sample.ndim == 2:
            # grayscale HxW
            pass
        elif sample.ndim == 3:
            if sample.shape[2] >= 3 and self.channels == 1:
                # RGB -> luminance
                rgb = sample[:, :, :3].astype('float32')
                sample = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
            elif sample.shape[2] >= 3 and self.channels == 3:
                sample = sample[:, :, :3]
            elif sample.shape[2] == 1:
                sample = sample[:, :, 0]
            else:
                raise ValueError(f"Unsupported image shape {sample.shape} for {path}")
        else:
            raise ValueError(f"Unsupported image ndim {sample.ndim} for {path}")

        sample = np.clip(sample, 0, data_range).astype('float32') / data_range

        if self.channels == 1:
            if sample.ndim == 2:
                sample = sample[:, :, None]
            elif sample.ndim == 3 and sample.shape[2] != 1:
                raise ValueError(f"Expected 1 channel, got shape {sample.shape} for {path}")
        else:
            if sample.ndim == 2:
                sample = np.stack([sample, sample, sample], axis=-1)
            elif sample.shape[2] != 3:
                raise ValueError(f"Expected 3 channels, got shape {sample.shape} for {path}")

        return sample

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

    def random_crop_size(self, imageA, imageB, crop_size):
        imageA = self.resize_shape(imageA, crop_size)
        imageB = self.resize_shape(imageB, crop_size)

        if not self.check_size(imageA, imageB):
            exit('Unmatched image sizes')

        h, w = imageB.shape[0], imageB.shape[1]
        h_start = np.random.randint(0, h - crop_size + 1)
        w_start = np.random.randint(0, w - crop_size + 1)
        imageA_crop = imageA[h_start:h_start + crop_size, w_start:w_start + crop_size, :]
        imageB_crop = imageB[h_start:h_start + crop_size, w_start:w_start + crop_size, :]
        return imageA_crop, imageB_crop

    def random_dihedral_augment(self, imageA, imageB):
        """Apply a random D4 symmetry (k*90° rotation and/or reflection) to both images."""
        k = np.random.randint(0, 4)
        do_flip = np.random.rand() < 0.5

        def _apply(img):
            out = np.rot90(img, k=k, axes=(0, 1))
            if do_flip:
                out = np.flip(out, axis=1)  # horizontal reflection
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
    print('Hello World')
    set_seed(11)
    root = os.path.join(os.path.dirname(__file__), '..', 'radiography', 'fix_test_v2', 'dataset.json')
    for mode in ['train', 'test']:
        ds = RSDataset(dataset_json=root, mode=mode, channels=1)
        print(len(ds))
        dataloader = DataLoader(ds, batch_size=1, pin_memory=True, shuffle=True)
        for i, (lq, hq, paths) in enumerate(dataloader):
            print(f"{mode} {paths[0]}: hq shape={hq.shape}, lq shape={lq.shape}")
            if i >= 2:
                break
