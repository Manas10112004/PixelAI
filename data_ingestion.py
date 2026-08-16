"""
data_ingestion.py
==================
Turns the two contest archives -- `Train.zip` and `Test_NoisyLR.zip` -- into
PyTorch-ready datasets with automated layout auto-detection.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split

_LR_HINTS = ("lr", "noisy", "degraded", "input", "low")
_HR_HINTS = ("hr", "clean", "gt", "target", "ground_truth", "groundtruth", "high")

_ARRAY_EXT = ".npy"
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
_VALID_EXTS = (_ARRAY_EXT,) + _IMAGE_EXTS


def extract_zip(zip_path: str, extract_dir: str, force: bool = False) -> str:
    if os.path.isdir(extract_dir) and os.listdir(extract_dir) and not force:
        return extract_dir
    if force and os.path.isdir(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            base = os.path.basename(member)
            if member.startswith("__MACOSX/") or base.startswith("._"):
                continue
            if member.endswith("/"):
                continue
            target_path = os.path.join(extract_dir, member)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with zf.open(member) as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return extract_dir


def load_array(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == _ARRAY_EXT:
        arr = np.load(path).astype(np.float32)
        return arr
    else:
        from PIL import Image
        img = Image.open(path)
        arr = np.array(img).astype(np.float32) / 255.0
        return arr


def _to_chw_tensor(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        if arr.shape[0] not in (1, 3) and arr.shape[2] in (1, 3):
            arr = arr.transpose(2, 0, 1)
    else:
        raise ValueError(f"Unsupported array rank {arr.ndim} for shape {arr.shape}")
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


def _list_samples(folder: str) -> List[str]:
    out = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in _VALID_EXTS:
                out.append(os.path.join(root, f))
    return sorted(out)


@dataclass
class DetectedLayout:
    mode: str
    lr_dir: Optional[str] = None
    hr_dir: Optional[str] = None
    pool_dir: Optional[str] = None
    pairs: List[Tuple[str, str]] = field(default_factory=list)
    pool_files: List[str] = field(default_factory=list)


def detect_train_layout(train_root: str) -> DetectedLayout:
    all_dirs = []
    for root, dirs, _files in os.walk(train_root):
        for d in dirs:
            all_dirs.append(os.path.join(root, d))

    def _hint_match(name: str, hints: Tuple[str, ...]) -> bool:
        low = name.lower()
        return any(h in low for h in hints)

    lr_dir, hr_dir = None, None
    for d in all_dirs:
        name = os.path.basename(d)
        if lr_dir is None and _hint_match(name, _LR_HINTS):
            lr_dir = d
        if hr_dir is None and _hint_match(name, _HR_HINTS):
            hr_dir = d

    if lr_dir and hr_dir:
        lr_files = {os.path.splitext(os.path.basename(p))[0]: p for p in _list_samples(lr_dir)}
        hr_files = {os.path.splitext(os.path.basename(p))[0]: p for p in _list_samples(hr_dir)}
        common = sorted(set(lr_files) & set(hr_files))
        if common:
            pairs = [(lr_files[k], hr_files[k]) for k in common]
            return DetectedLayout(mode="paired", lr_dir=lr_dir, hr_dir=hr_dir, pairs=pairs)

    pool_files = _list_samples(train_root)
    return DetectedLayout(mode="unpaired", pool_dir=train_root, pool_files=pool_files)


class PairedRealDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], crop_size: int = 128, train: bool = True):
        self.pairs = pairs
        self.crop_size = crop_size - (crop_size % 16)
        self.train = train
        if len(pairs) == 0:
            raise ValueError("PairedRealDataset received zero pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def _rand_crop_pair(self, lr: torch.Tensor, hr: torch.Tensor, scale: int):
        _, h, w = lr.shape
        cs = min(self.crop_size, h, w)
        cs = cs - (cs % 16)
        if cs < 16:
            return lr, hr

        top = 0 if h == cs else int(torch.randint(0, h - cs, (1,)).item())
        left = 0 if w == cs else int(torch.randint(0, w - cs, (1,)).item())
        lr_c = lr[:, top:top + cs, left:left + cs]
        hr_c = hr[:, top * scale:(top + cs) * scale, left * scale:(left + cs) * scale]
        return lr_c, hr_c

    def __getitem__(self, idx: int):
        lr_path, hr_path = self.pairs[idx]
        lr = _to_chw_tensor(load_array(lr_path))
        hr = _to_chw_tensor(load_array(hr_path))

        scale = round(hr.shape[-1] / lr.shape[-1])
        scale = max(scale, 1)

        if self.train:
            lr, hr = self._rand_crop_pair(lr, hr, scale)
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                lr, hr = torch.rot90(lr, k, dims=(1, 2)), torch.rot90(hr, k, dims=(1, 2))
            if torch.rand(1).item() < 0.5:
                lr, hr = torch.flip(lr, dims=(2,)), torch.flip(hr, dims=(2,))
            if torch.rand(1).item() < 0.5:
                lr, hr = torch.flip(lr, dims=(1,)), torch.flip(hr, dims=(1,))

        return lr.contiguous(), hr.contiguous(), torch.tensor(scale, dtype=torch.long)


class NpyInferenceDataset(Dataset):
    def __init__(self, folder: str):
        self.files = _list_samples(folder)
        if len(self.files) == 0:
            raise FileNotFoundError(f"No samples found under {folder}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.files[idx]
        arr = load_array(path)
        tensor = _to_chw_tensor(arr)
        stem = os.path.splitext(os.path.basename(path))[0]
        return tensor, stem


def prepare_datasets(
    train_zip: str,
    test_zip: str,
    work_dir: str = "./data_ingested",
    crop_size: int = 128,
    val_split: float = 0.1,
    seed: int = 42,
):
    train_root = extract_zip(train_zip, os.path.join(work_dir, "train_raw"))
    test_root = extract_zip(test_zip, os.path.join(work_dir, "test_raw"))

    layout = detect_train_layout(train_root)

    if layout.mode == "paired":
        full_ds = PairedRealDataset(layout.pairs, crop_size=crop_size, train=True)
        sample_arr = load_array(layout.pairs[0][0])
        sample_hr = load_array(layout.pairs[0][1])
        sr_scale = max(1, round(sample_hr.shape[-1] / sample_arr.shape[-1]))
        n_pairs = len(layout.pairs)
    else:
        from dataset import SemiconHPCDataset, PatchConfig
        full_ds = SemiconHPCDataset(
            layout.pool_dir,
            patch_cfg=PatchConfig(crop_size=crop_size),
            train=True,
        )
        sample_arr = load_array(layout.pool_files[0])
        n_pairs = len(layout.pool_files)
        sr_scale = 1

    in_ch = 1 if sample_arr.ndim == 2 or (sample_arr.ndim == 3 and sample_arr.shape[-1] == 1) else sample_arr.shape[-1]

    n_val = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)

    test_ds = NpyInferenceDataset(test_root)

    meta = {
        "train_zip": train_zip,
        "test_zip": test_zip,
        "detected_mode": layout.mode,
        "in_channels": in_ch,
        "sr_scale": sr_scale,
        "n_train_source_samples": n_pairs,
        "n_train_split": n_train,
        "n_val_split": n_val,
        "n_test_samples": len(test_ds),
        "crop_size": crop_size,
    }
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, "ingestion_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return train_ds, val_ds, test_ds, meta