import glob
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image

def ultimate_cv2_rescue(arr: np.ndarray) -> np.ndarray:
    """The exact OpenCV rescue filter used in your submission."""
    if arr.max() <= 1.0:
        arr = arr * 255.0
    img_uint8 = np.clip(arr, 0, 255).astype(np.uint8)

    is_3d = (img_uint8.ndim == 3 and img_uint8.shape[0] in [1, 3])
    if is_3d:
        img_uint8 = img_uint8.transpose(1, 2, 0)

    # 1. Crush Salt & Pepper Noise
    median_filtered = cv2.medianBlur(img_uint8, 5)

    # 2. Crush Gaussian/Speckle Noise
    if median_filtered.ndim == 3 and median_filtered.shape[-1] == 3:
        denoised = cv2.fastNlMeansDenoisingColored(median_filtered, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    else:
        denoised = cv2.fastNlMeansDenoising(median_filtered, None, h=10, templateWindowSize=7, searchWindowSize=21)

    out_float = denoised.astype(np.float32) / 255.0

    if is_3d:
        if out_float.ndim == 2:
            out_float = np.expand_dims(out_float, axis=-1)
        out_float = out_float.transpose(2, 0, 1)

    return out_float

def main():
    # Make search recursive to find files no matter how the zip extracted
    gt_files = glob.glob("./data_ingested/train_raw/**/*.png", recursive=True) + \
               glob.glob("./data_ingested/train_raw/**/*.npy", recursive=True)
    
    if not gt_files:
        print("No Ground Truth images found! Using fallback metrics for presentation.")
        return
        
    print(f"[*] Calculating metrics on {min(5, len(gt_files))} sample(s)...")
    
    psnr_scores = []
    ssim_scores = []
    
    for f in gt_files[:5]:
        if f.endswith('.npy'):
            gt_img = np.load(f).astype(np.float32)
            if gt_img.max() > 1.0: gt_img /= 255.0
            if gt_img.ndim == 3 and gt_img.shape[0] in [1, 3]:
                gt_img = gt_img.transpose(1, 2, 0)
            if gt_img.ndim == 3:
                gt_img = gt_img.mean(axis=-1)
        else:
            gt_img = np.array(Image.open(f).convert("L"), dtype=np.float32) / 255.0
            
        # Inject synthetic noise
        noisy = gt_img + np.random.normal(0, 0.12, gt_img.shape).astype(np.float32)
        prob = 0.08
        mask = np.random.rand(*gt_img.shape)
        noisy[mask < (prob / 2.0)] = 0.0
        noisy[(mask >= (prob / 2.0)) & (mask < prob)] = 1.0
        noisy = np.clip(noisy, 0.0, 1.0)
        
        restored = ultimate_cv2_rescue(noisy)
        
        p = psnr(gt_img, restored, data_range=1.0)
        s = ssim(gt_img, restored, data_range=1.0)
        
        psnr_scores.append(p)
        ssim_scores.append(s)
        
    print("\n" + "="*40)
    print("📊 METRICS FOR SLIDE 6")
    print("="*40)
    print(f"Average PSNR : {np.mean(psnr_scores):.2f} dB")
    print(f"Average SSIM : {np.mean(ssim_scores):.4f}")
    print("="*40)

if __name__ == "__main__":
    main()