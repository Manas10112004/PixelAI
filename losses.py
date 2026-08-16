"""
losses.py
=========
Enhanced multi-component loss for image restoration targeting impulse/salt-and-pepper 
noise removal, edge preservation, and structural consistency.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImpulseDenoisingLoss(nn.Module):
    def __init__(
        self,
        charbonnier_eps: float = 1e-3,
        l1_weight: float = 2.0,
        sobel_weight: float = 1.0,
        fft_weight: float = 0.1,
    ):
        super().__init__()
        self.eps2 = charbonnier_eps ** 2
        self.l1_weight = l1_weight
        self.sobel_weight = sobel_weight
        self.fft_weight = fft_weight

        # 3x3 Sobel Filters for Gradient Tracking
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        ky = kx.transpose(2, 3)
        self.register_buffer("kx", kx, persistent=False)
        self.register_buffer("ky", ky, persistent=False)

    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        b, c, h, w = x_f.shape
        x_flat = x_f.reshape(b * c, 1, h, w)
        kx = self.kx.to(device=x_f.device, dtype=x_f.dtype)
        ky = self.ky.to(device=x_f.device, dtype=x_f.dtype)
        gx = F.conv2d(x_flat, kx, padding=1)
        gy = F.conv2d(x_flat, ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        return mag.reshape(b, c, h, w)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
    ):
        # 1. Direct L1 Loss (High penalty on sparse, high-amplitude salt-and-pepper noise spikes)
        l1_loss = F.l1_loss(pred, target)

        # 2. Charbonnier Smooth Loss (Overall smooth reconstruction)
        diff = pred - target
        l_char = torch.mean(torch.sqrt(diff * diff + self.eps2))

        # 3. Sobel Edge Gradient Loss (Preserves high-frequency image contours)
        pred_grad = self._gradient_magnitude(pred)
        target_grad = self._gradient_magnitude(target)
        l_sobel = F.l1_loss(pred_grad, target_grad)

        # 4. FFT Frequency Spectrum Loss
        pred_fft = torch.fft.fft2(pred.float(), norm="ortho")
        target_fft = torch.fft.fft2(target.float(), norm="ortho")
        l_fft = torch.mean(torch.abs(pred_fft - target_fft))

        # Total Composite Loss
        total = (
            self.l1_weight * l1_loss
            + l_char
            + self.sobel_weight * l_sobel
            + self.fft_weight * l_fft
        )

        if return_components:
            components = {
                "l1": l1_loss.detach(),
                "charbonnier": l_char.detach(),
                "sobel": l_sobel.detach(),
                "fft": l_fft.detach(),
                "total": total.detach(),
            }
            return total, components
        return total


# Backward compatibility mapping for existing train.py references
SemiconRestorationLoss = ImpulseDenoisingLoss


if __name__ == "__main__":
    loss_fn = ImpulseDenoisingLoss()
    pred = torch.rand(4, 1, 256, 256, requires_grad=True)
    target = torch.rand(4, 1, 256, 256)
    total, comps = loss_fn(pred, target, return_components=True)
    total.backward()
    print("Loss initialization check passed.")
    print("Total Loss:", total.item())
    print("Components:", {k: round(v.item(), 5) for k, v in comps.items()})