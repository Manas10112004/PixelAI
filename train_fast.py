import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from architecture import SemiconNAFNet

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class FastSemiconDataset(Dataset):
    def __init__(self, patch_size=256, max_samples=100):
        super().__init__()
        self.patch_size = patch_size
        
        search_paths = [
            "./data_ingested/train_raw/train/GT/*.png",
            "./data_ingested/train_raw/train/GT/*.npy",
            "./train_data/*.png",
            "./train_data/*.npy",
            "./*.png"
        ]
        
        files = []
        for path in search_paths:
            files.extend(glob.glob(path))
        files = list(set(files))[:max_samples]
        
        print(f"[DataLoader] Caching {len(files)} samples into RAM...")
        self.cached_gt = []
        
        for f in files:
            try:
                if f.endswith(".npy"):
                    img = np.load(f).astype(np.float32)
                    if img.max() > 1.0: img /= 255.0
                    if img.ndim == 2: img = np.expand_dims(img, axis=0)
                else:
                    from PIL import Image
                    pil_img = Image.open(f).convert("L")
                    img = np.expand_dims(np.array(pil_img, dtype=np.float32) / 255.0, axis=0)

                if img.shape[0] == 1:
                    img = np.repeat(img, 3, axis=0)

                self.cached_gt.append(torch.from_numpy(img))
            except Exception:
                continue

        print(f"[✓] {len(self.cached_gt)} images cached.")

    def __len__(self):
        return len(self.cached_gt)

    def __getitem__(self, idx):
        img = self.cached_gt[idx]
        _, h, w = img.shape
        ps = self.patch_size

        if h >= ps and w >= ps:
            top = torch.randint(0, h - ps + 1, (1,)).item()
            left = torch.randint(0, w - ps + 1, (1,)).item()
            gt = img[:, top:top+ps, left:left+ps]
        else:
            gt = F.interpolate(img.unsqueeze(0), size=(ps, ps), mode='bilinear', align_corners=False).squeeze(0)

        noisy = gt + torch.randn_like(gt) * 0.10
        noisy = torch.clamp(noisy, 0.0, 1.0)
        
        mask = torch.rand_like(noisy)
        noisy[mask < 0.04] = 0.0
        noisy[(mask >= 0.04) & (mask < 0.08)] = 1.0

        noisy_np = (noisy.numpy() * 255.0).astype(np.uint8)
        for c in range(3):
            noisy_np[c] = cv2.medianBlur(noisy_np[c], 3)
        
        prefiltered_noisy = torch.from_numpy(noisy_np.astype(np.float32) / 255.0)
        return prefiltered_noisy, gt

def run_fast_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Fast FP32 Denoising Engine on {device}...")

    os.makedirs("./checkpoints", exist_ok=True)
    dataset = FastSemiconDataset(patch_size=256, max_samples=100)
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    model = SemiconNAFNet().to(device)
    criterion = nn.L1Loss().to(device)

    # Conservative learning rate to prevent exploding gradients
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    epochs = 10
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for noisy, gt in loader:
            noisy, gt = noisy.to(device), gt.to(device)
            optimizer.zero_grad(set_to_none=True)

            out = model(noisy)
            loss = criterion(out, gt)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Epoch [{epoch+1:02d}/{epochs}] - Loss: {avg_loss:.5f}")

        if avg_loss < best_loss and not np.isnan(avg_loss):
            best_loss = avg_loss
            torch.save({"model_state_dict": model.state_dict()}, "./checkpoints/best_model.pt")
            print(f"  [✓] Checkpoint saved (Loss: {best_loss:.5f})")

    print("\n[✓] Fast training complete! Model weights updated successfully.")

if __name__ == "__main__":
    run_fast_training()