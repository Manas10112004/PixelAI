"""
evaluate_metrics.py
-------------------
Computes quantitative restoration metrics (PSNR, SSIM, MSE) between
the degraded input image and the restored model output.
"""

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def compute_metrics(original_path: str, restored_path: str):
    img_orig = np.array(Image.open(original_path).convert("L"))
    img_rest = np.array(Image.open(restored_path).convert("L"))

    mse_val = np.mean((img_orig.astype(np.float64) - img_rest.astype(np.float64)) ** 2)
    psnr_val = psnr(img_orig, img_rest, data_range=255)
    ssim_val = ssim(img_orig, img_rest, data_range=255)

    print("=" * 45)
    print("      QUANTITATIVE RESTORATION ANALYSIS      ")
    print("=" * 45)
    print(f"Mean Squared Error (MSE) : {mse_val:.6f}")
    print(f"PSNR (dB)                : {psnr_val:.4f} dB")
    print(f"SSIM                     : {ssim_val:.4f}")
    print("=" * 45)

    if mse_val < 1e-4:
        print("[!] NOTICE: Output image is nearly identical to input image.")
        print("    Check if model weights in 'best_model.pt' are fully trained.")
    else:
        print("[✓] Model has actively modified the image pixels.")

if __name__ == "__main__":
    compute_metrics("t1.png", "restored_t1.png")