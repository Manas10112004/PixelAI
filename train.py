import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from architecture import SemiconNAFNet
from dataset import SemiconDataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class CombinedDenoisingLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()

        # Direct L1 Loss aggressively targets salt and pepper spikes
        l1 = F.l1_loss(pred, target)
        
        # Numerically Safe 2D FFT Frequency Loss
        fft_pred = torch.fft.rfft2(pred)
        fft_target = torch.fft.rfft2(target)
        diff = fft_pred - fft_target
        
        # Use safe magnitude sqrt(r^2 + i^2 + eps) to prevent NaN gradients
        fft_loss = torch.mean(torch.sqrt(diff.real**2 + diff.imag**2 + 1e-8))

        return l1 + 0.05 * fft_loss

def run_training():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting High-Speed Denoising Engine on {device}...")

    os.makedirs("./checkpoints", exist_ok=True)

    dataset = SemiconDataset(patch_size=256)
    
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,  # Prevents Windows multiprocessing locks
        pin_memory=True,
        drop_last=True
    )

    model = SemiconNAFNet().to(device)
    criterion = CombinedDenoisingLoss().to(device)

    # Reduced learning rate slightly for stability
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 150
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    scaler = torch.amp.GradScaler('cuda')
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for noisy, gt in loader:
            noisy, gt = noisy.to(device, non_blocking=True), gt.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                out = model(noisy)
                
            # Disable autocast during loss computation to avoid FP16 FFT instability
            with torch.amp.autocast('cuda', enabled=False):
                loss = criterion(out, gt)

            # Check for NaN before scaling
            if torch.isnan(loss) or torch.isinf(loss):
                print(f" Warning: NaN detected at Epoch {epoch+1}, skipping step.")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(loader)
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1:03d}/{epochs}] - Loss: {avg_loss:.5f} - LR: {lr:.6f}")

        if avg_loss < best_loss and not torch.isnan(torch.tensor(avg_loss)):
            best_loss = avg_loss
            torch.save({"model_state_dict": model.state_dict()}, "./checkpoints/best_model.pt")
            print(f"  [✓] Model checkpoint saved at epoch {epoch+1} (Best Loss: {best_loss:.5f})")

if __name__ == "__main__":
    run_training()