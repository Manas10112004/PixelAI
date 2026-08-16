import argparse
import os
import numpy as np
import torch
from architecture import SemiconNAFNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SemiconNAFNet().to(device)
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()

    if args.input.endswith(".npy"):
        img = np.load(args.input).astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0
        is_single_chan = (img.ndim == 2 or (img.ndim == 3 and img.shape[0] == 1))
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)
    else:
        from PIL import Image
        pil_img = Image.open(args.input).convert("RGB")
        img = np.array(pil_img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        is_single_chan = False

    inp_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out_tensor = model(inp_tensor)

    out_img = out_tensor.squeeze(0).cpu().numpy()
    out_img = np.clip(out_img, 0.0, 1.0)

    # Transpose channels to (H, W, C) for image formatting
    if out_img.ndim == 3 and out_img.shape[0] in [1, 3]:
        out_img = np.transpose(out_img, (1, 2, 0))

    if is_single_chan:
        out_img = out_img.mean(axis=-1)

    # Ensure 2D array if single channel
    out_img = np.squeeze(out_img)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.output.endswith(".npy"):
        np.save(args.output, out_img)
    else:
        from PIL import Image
        out_uint8 = (out_img * 255.0).astype(np.uint8)
        Image.fromarray(out_uint8).save(args.output)
    
    print(f"Successfully generated restored image at {args.output}")

if __name__ == "__main__":
    main()