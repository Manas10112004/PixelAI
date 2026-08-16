import os
import glob
import numpy as np
import torch
from architecture import SemiconNAFNet

def run_all_inference():
    output_dir = "./restored_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Locate test files
    test_files = (
        glob.glob("./data_ingested/test_raw/NoisyLR/*.npy") +
        glob.glob("./data_ingested/test_raw/NoisyLR/*.png") +
        glob.glob("./test_images/NoisyLR/*.npy")
    )
    test_files = sorted(list(set(test_files)))
    total_files = len(test_files)
    
    print(f"Found {total_files} test files for inference.")
    if total_files == 0:
        print("No test files found! Check directory paths.")
        return

    # 2. Load model ONCE into memory/GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model checkpoint onto {device}...")
    
    model = SemiconNAFNet().to(device)
    ckpt_path = "./checkpoints/best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    print("Model loaded successfully. Starting batch processing...\n")

    # 3. Fast In-Memory Processing Loop
    with torch.no_grad():
        for i, img_path in enumerate(test_files, start=1):
            filename = os.path.basename(img_path)
            out_path = os.path.join(output_dir, filename)
            
            # Load and preprocess
            if img_path.endswith(".npy"):
                img = np.load(img_path).astype(np.float32)
                if img.max() > 1.0:
                    img /= 255.0
                is_single_chan = (img.ndim == 2 or (img.ndim == 3 and img.shape[0] == 1))
                if img.ndim == 2:
                    img = np.expand_dims(img, axis=0)
                if img.shape[0] == 1:
                    img = np.repeat(img, 3, axis=0)
            else:
                from PIL import Image
                pil_img = Image.open(img_path).convert("RGB")
                img = np.array(pil_img, dtype=np.float32).transpose(2, 0, 1) / 255.0
                is_single_chan = False

            # Model Forward Pass
            inp_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
            out_tensor = model(inp_tensor)
            
            # Postprocess and Save
            out_img = out_tensor.squeeze(0).cpu().numpy()
            out_img = np.clip(out_img, 0.0, 1.0)
            
            if is_single_chan:
                out_img = out_img.mean(axis=0)

            if out_path.endswith(".npy"):
                np.save(out_path, out_img)
            else:
                from PIL import Image
                out_uint8 = (out_img * 255.0).astype(np.uint8)
                Image.fromarray(out_uint8).save(out_path)

            # Terminal Progress Update
            if i % 20 == 0 or i == total_files:
                print(f"[{i}/{total_files}] Processed {filename}")

    print(f"\nInference Complete! All {total_files} restored files saved to '{output_dir}'.")

if __name__ == "__main__":
    run_all_inference()