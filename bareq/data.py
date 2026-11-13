"""Training samples and val/test image lists.

Every training sample is a pure function of (seed, index): the crop, the flips and the synthetic degradation
all come from one generator seeded with both. So a resumed run sees exactly the data it would have seen.
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

RUN_TASKS = {"dark": ["dark"], "noisy": ["noisy"], "old": ["old"], "all": ["dark", "noisy", "old"]}
RUN_FILES = {
    "dark": ["train_dark_low.npy", "train_dark_high.npy"],
    "noisy": ["train_sidd_noisy.npy", "train_sidd_gt.npy", "train_clean.npy"],
    "old": ["train_clean.npy"],
}
RUN_FILES["all"] = sorted(set(sum(RUN_FILES.values(), [])))


class TrainSet(Dataset):
    """Balanced by task: sample i belongs to tasks[i % len(tasks)], so in the mixed run each task gets a third.
    The noisy task is half real (SIDD) and half synthetic noise on clean photos."""

    def __init__(self, data_dir, run, crop, seed, length, offset=0):
        self.dir, self.run, self.crop, self.seed = data_dir, run, crop, seed
        self.length, self.offset = length, offset
        self.tasks = RUN_TASKS[run]
        self.arr = None

    def _open(self):
        import cv2
        cv2.setNumThreads(0)
        self.arr = {f[6:-4]: np.load(os.path.join(self.dir, f), mmap_mode="r") for f in RUN_FILES[self.run]}

    def __len__(self):
        return self.length - self.offset

    def _crop_pair(self, a, b, rng):
        n, h, w, _ = a.shape
        i = int(rng.integers(n))
        y, x = int(rng.integers(h - self.crop + 1)), int(rng.integers(w - self.crop + 1))
        s = np.s_[i, y:y + self.crop, x:x + self.crop]
        return np.array(a[s]), (np.array(b[s]) if b is not None else None)

    def __getitem__(self, k):
        from . import degrade
        if self.arr is None:
            self._open()
        idx = self.offset + k
        rng = np.random.default_rng([self.seed, idx])
        task = self.tasks[idx % len(self.tasks)]
        A = self.arr
        if task == "dark":
            x, y = self._crop_pair(A["dark_low"], A["dark_high"], rng)
        elif task == "noisy" and rng.random() < 0.5:
            x, y = self._crop_pair(A["sidd_noisy"], A["sidd_gt"], rng)
        else:
            y, _ = self._crop_pair(A["clean"], None, rng)
            fn = degrade.old if task == "old" else degrade.noisy
            x = degrade.to_u8(fn(y.astype(np.float32) / 255, rng)[0])
        # same flip / rotation for input and target
        if rng.random() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]
        if rng.random() < 0.5:
            x, y = x[::-1], y[::-1]
        r = int(rng.integers(4))
        x, y = np.rot90(x, r), np.rot90(y, r)
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)
        return to_t(x), to_t(y)  # uint8; scaled to [0, 1] on the GPU


def list_pairs(root, split, task, limit=0):
    d = os.path.join(root, split, task)
    names = sorted(os.listdir(os.path.join(d, "input")))
    if limit:  # spread over the list, not just the first few
        names = [names[i] for i in np.linspace(0, len(names) - 1, min(limit, len(names))).round().astype(int)]
    return [(n, os.path.join(d, "input", n), os.path.join(d, "target", n)) for n in names]


def load_image(path, device):
    a = np.array(Image.open(path).convert("RGB"))
    return torch.from_numpy(a).to(device).permute(2, 0, 1)[None].float() / 255
