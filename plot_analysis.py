import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_restoration_analysis(input_path="t1.png", restored_path="restored_t1.png", save_path="analysis_graph.png"):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if not os.path.exists(restored_path):
        raise FileNotFoundError(f"Restored file not found: {restored_path}. Run restore_image.py first!")

    img_in = np.array(Image.open(input_path).convert("L")).astype(np.float32)
    img_out = np.array(Image.open(restored_path).convert("L")).astype(np.float32)

    residual = np.abs(img_in - img_out) * 5.0

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.title("Input Image (Degraded)")
    plt.imshow(img_in, cmap="gray", vmin=0, vmax=255)
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.title("Model Output (Restored)")
    plt.imshow(img_out, cmap="gray", vmin=0, vmax=255)
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title("Amplified Residual Map (|Input - Output| x 5)")
    plt.imshow(residual, cmap="inferno", vmin=0, vmax=255)
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title("Pixel Intensity Distribution Histogram")
    plt.hist(img_in.ravel(), bins=256, range=(0, 255), alpha=0.5, color='red', label='Input')
    plt.hist(img_out.ravel(), bins=256, range=(0, 255), alpha=0.5, color='blue', label='Restored')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_restoration_analysis("t1.png", "restored_t1.png")