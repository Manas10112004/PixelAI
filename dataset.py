import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

class SemiconDataset(Dataset):
    def __init__(self, data_dir=None, patch_size=256):
        super().__init__()
        self.patch_size = patch_size
        
        search_paths = [
            "./data_ingested/train_raw/train/GT/*.png",
            "./data_ingested/train_raw/train/GT/*.npy",
            "./train_data/*.png",
            "./train_data/*.npy",
        ]
        if data_dir:
            search_paths.insert(0, os.path.join(data_dir, "*.png"))

        files = []
        for path in search_paths:
            files.extend(glob.glob(path))
        self.files = list(set(files))
        
        print(f"Pre-loading {len(self.files)} images into memory...")
        self.cached_tensors = []
        
        for f in self.files:
            try:
                if f.endswith(".npy"):
                    img = np.load(f).astype(np.float32)
                    if img.max() > 1.0:
                        img /= 255.0
                    if img.ndim == 2:
                        img = np.expand_dims(img, axis=0)
                else:
                    pil_img = Image.open(f).convert("L")
                    img = np.expand_dims(np.array(pil_img, dtype=np.float32) / 255.0, axis=0)

                if img.shape[0] == 1:
                    img = np.repeat(img, 3, axis=0)

                t = torch.from_numpy(img)
                self.cached_tensors.append(t)
            except Exception:
                continue

        print(f" Successfully cached {len(self.cached_tensors)} tensors in RAM.")

    def _apply_impulse_noise(self, x: torch.Tensor, prob: float = 0.08) -> torch.Tensor:
        mask = torch.rand_like(x)
        noisy = x.clone()
        noisy[mask < (prob / 2.0)] = 0.0
        noisy[(mask >= (prob / 2.0)) & (mask < prob)] = 1.0
        return noisy

    def __len__(self):
        return len(self.cached_tensors)

    def __getitem__(self, idx):
        img = self.cached_tensors[idx]
        _, h, w = img.shape
        ps = self.patch_size

        if h >= ps and w >= ps:
            top = torch.randint(0, h - ps + 1, (1,)).item()
            left = torch.randint(0, w - ps + 1, (1,)).item()
            gt = img[:, top:top+ps, left:left+ps]
        else:
            gt = F.interpolate(img.unsqueeze(0), size=(ps, ps), mode='bilinear', align_corners=False).squeeze(0)

        noisy = gt + torch.randn_like(gt) * 0.12
        noisy = torch.clamp(noisy, 0.0, 1.0)
        noisy = self._apply_impulse_noise(noisy, prob=0.08)

        return noisy, gt