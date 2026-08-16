"""
visualize.py
------------
Loads a single noisy image, runs inference using best_model.pt,
and displays/saves a side-by-side comparison (Noisy Input vs Restored Output).
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF

from architecture import SemiconNAFNet


def restore_single_image(image_path: str, checkpoint_path: str, device: str = "cuda"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    in_channels = ckpt.get("in_channels", 3)
    sr_scale = ckpt.get("sr_scale", 1)

    model = SemiconNAFNet(in_ch=in_channels, base_ch=32, sr_scale=sr_scale).to(device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    img_pil = Image.open(image_path).convert("L" if in_channels == 1 else "RGB")
    input_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
            output_tensor = model(input_tensor)

    output_tensor = output_tensor.squeeze(0).float().cpu().clamp(0.0, 1.0)
    if in_channels == 1:
        output_np = output_tensor.squeeze(0).numpy()
        restored_pil = Image.fromarray((output_np * 255.0).astype(np.uint8), mode="L")
    else:
        output_np = output_tensor.permute(1, 2, 0).numpy()
        restored_pil = Image.fromarray((output_np * 255.0).astype(np.uint8), mode="RGB")

    return img_pil, restored_pil


def main():
    parser = argparse.ArgumentParser(description="Visualize image restoration result")
    parser.add_argument("--image", type=str, required=True, help="Path to degraded/noisy input image")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt", help="Path to checkpoint file")
    parser.add_argument("--save", type=str, default="comparison_result.png", help="Path to save output comparison plot")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running inference on {args.image} using device: {device}...")

    noisy_img, restored_img = restore_single_image(args.image, args.checkpoint, device=device)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(noisy_img, cmap="gray" if noisy_img.mode == "L" else None)
    axes[0].set_title("Input (Noisy / Degraded)", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(restored_img, cmap="gray" if restored_img.mode == "L" else None)
    axes[1].set_title("Model Output (Restored)", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(args.save, dpi=300, bbox_inches="tight")
    print(f"Comparison saved to: {args.save}")

    plt.show()


if __name__ == "__main__":
    main()