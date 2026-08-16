"""
eval_hpc.py
===========
Standalone contest-runner evaluation script: tiled, Gaussian-weighted,
CUDA-streamed inference over arbitrarily large wafer-scan images, plus
ONNX/TensorRT export helpers for deployment.

Run (inference):
    python eval_hpc.py infer --checkpoint best_model.pt \
        --input_dir /path/to/degraded --output_dir /path/to/restored

Run (export):
    python eval_hpc.py export --checkpoint best_model.pt \
        --onnx_path model.onnx
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from architecture import SemiconNAFNet
from data_ingestion import extract_zip, load_array, _to_chw_tensor, _list_samples


# --------------------------------------------------------------------------- #
# Gaussian tile-blending window
# --------------------------------------------------------------------------- #

def _make_gaussian_window(tile_size: int, sigma_ratio: float = 0.5) -> torch.Tensor:
    """
    2D separable Gaussian weighting window used to blend overlapping tile
    predictions. Weights peak at the tile center and decay toward the
    edges, so seam lines at tile boundaries are smoothly averaged away
    rather than producing hard cut discontinuities.
    """
    sigma = tile_size * sigma_ratio / 2.0
    coords = torch.arange(tile_size, dtype=torch.float32) - (tile_size - 1) / 2.0
    g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    window_2d = torch.outer(g1d, g1d)
    window_2d = window_2d / window_2d.max()  # normalize peak to 1.0
    return window_2d


# --------------------------------------------------------------------------- #
# Tiling engine
# --------------------------------------------------------------------------- #

class TiledInferenceEngine:
    """
    Processes arbitrarily large images by splitting into overlapping tiles,
    running the model on each tile (batched where memory allows), and
    stitching results back together with Gaussian-weighted blending.

    HPC notes:
    - Tiles are accumulated into a float32 CUDA "canvas" + a matching weight
      canvas; both live entirely in GPU memory for the duration of one
      image, avoiding host round-trips between tiles.
    - A dedicated `torch.cuda.Stream` is used for the copy-in of the next
      tile while the current tile's forward pass runs on the default
      stream, overlapping data movement with compute.
    - Batches of tiles (not single tiles) are fed to the model when they
      fit in memory, to better saturate the GPU's Tensor Cores per launch.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tile_size: int = 512,
        overlap: int = 32,
        sr_scale: int = 2,
        tile_batch_size: int = 4,
        device: str = "cuda",
        amp_dtype: torch.dtype = torch.float16,
    ):
        self.model = model
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.sr_scale = sr_scale
        self.tile_batch_size = tile_batch_size
        self.device = device
        self.amp_dtype = amp_dtype

        self.window = _make_gaussian_window(tile_size).to(device)
        # Precompute the upsampled window once too (reused every image).
        self.window_up = _make_gaussian_window(tile_size * sr_scale).to(device)

        self.copy_stream = (
            torch.cuda.Stream() if device == "cuda" else None
        )

    def _tile_coords(self, h: int, w: int) -> List[Tuple[int, int]]:
        coords = []
        ys = list(range(0, max(h - self.tile_size, 0) + 1, self.stride))
        xs = list(range(0, max(w - self.tile_size, 0) + 1, self.stride))
        if not ys or ys[-1] != h - self.tile_size:
            ys.append(max(h - self.tile_size, 0))
        if not xs or xs[-1] != w - self.tile_size:
            xs.append(max(w - self.tile_size, 0))
        for y in ys:
            for x in xs:
                coords.append((y, x))
        return coords

    @torch.no_grad()
    def infer(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: (C, H, W) float tensor in [0, 1], on CPU or GPU.
        Returns restored (C, H*scale, W*scale) tensor on `device`.
        """
        c, h, w = image.shape
        ts = self.tile_size

        # Pad the source image if smaller than one tile.
        pad_h = max(ts - h, 0)
        pad_w = max(ts - w, 0)
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
            h, w = image.shape[-2], image.shape[-1]

        image = image.to(self.device, non_blocking=True)

        coords = self._tile_coords(h, w)
        out_h, out_w = h * self.sr_scale, w * self.sr_scale

        canvas = torch.zeros((c, out_h, out_w), device=self.device, dtype=torch.float32)
        weight_canvas = torch.zeros((1, out_h, out_w), device=self.device, dtype=torch.float32)

        # Process tiles in batches to keep Tensor Cores well-fed.
        for batch_start in range(0, len(coords), self.tile_batch_size):
            batch_coords = coords[batch_start: batch_start + self.tile_batch_size]

            # Overlap tile gather (copy) with previous batch's compute via a
            # side stream, then synchronize before the forward pass.
            if self.copy_stream is not None:
                with torch.cuda.stream(self.copy_stream):
                    tiles = torch.stack(
                        [image[:, y:y + ts, x:x + ts] for y, x in batch_coords], dim=0
                    )
                torch.cuda.current_stream().wait_stream(self.copy_stream)
            else:
                tiles = torch.stack(
                    [image[:, y:y + ts, x:x + ts] for y, x in batch_coords], dim=0
                )

            tiles = tiles.to(memory_format=torch.channels_last)

            with torch.cuda.amp.autocast(dtype=self.amp_dtype, enabled=(self.device == "cuda")):
                preds = self.model(tiles)

            preds = preds.float()

            for i, (y, x) in enumerate(batch_coords):
                oy, ox = y * self.sr_scale, x * self.sr_scale
                oh, ow = ts * self.sr_scale, ts * self.sr_scale
                weighted = preds[i] * self.window_up.unsqueeze(0)
                canvas[:, oy:oy + oh, ox:ox + ow] += weighted
                weight_canvas[:, oy:oy + oh, ox:ox + ow] += self.window_up.unsqueeze(0)

        weight_canvas = weight_canvas.clamp_min(1e-8)
        restored = canvas / weight_canvas

        # Crop off any reflect-padding (scaled by sr_scale).
        orig_out_h = (h - pad_h) * self.sr_scale
        orig_out_w = (w - pad_w) * self.sr_scale
        return restored[:, :orig_out_h, :orig_out_w].clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# High-level inference driver
# --------------------------------------------------------------------------- #

def load_model(
    checkpoint_path: str,
    device: str = "cuda",
    in_ch: int = None,
    sr_scale: int = None,
) -> torch.nn.Module:
    """
    Builds SemiconNAFNet and loads weights. `in_ch` / `sr_scale` are read
    from the checkpoint (saved by train.py alongside the weights) when not
    explicitly overridden -- this is what lets eval_hpc.py automatically
    match the 1-channel, scale-1 configuration that Train.zip /
    Test_NoisyLR.zip actually require, without the caller having to know
    those details up front.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    if in_ch is None:
        in_ch = ckpt.get("in_channels", 3) if isinstance(ckpt, dict) else 3
    if sr_scale is None:
        sr_scale = ckpt.get("sr_scale", 2) if isinstance(ckpt, dict) else 2

    model = SemiconNAFNet(in_ch=in_ch, base_ch=32, sr_scale=sr_scale).to(device)
    # Strip potential torch.compile "_orig_mod." prefix from compiled checkpoints.
    state_dict = { k.replace("_orig_mod.", ""): v for k, v in state_dict.items() }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(memory_format=torch.channels_last)
    return model, in_ch, sr_scale


def run_folder_inference(
    checkpoint_path: str,
    input_path: str,
    output_dir: str,
    tile_size: int = 512,
    overlap: int = 32,
    sr_scale: int = None,
    tile_batch_size: int = 4,
    save_png_preview: bool = True,
):
    """
    `input_path` may be either a directory of samples or a zip archive
    (e.g. Test_NoisyLR.zip itself) -- zips are auto-extracted into
    `output_dir/_extracted_input` first. Each sample is restored and saved
    as `.npy` (matching the contest's native format) plus, optionally, a
    normalized 8-bit PNG preview for quick visual sanity-checking.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    if input_path.lower().endswith(".zip"):
        input_dir = extract_zip(input_path, os.path.join(output_dir, "_extracted_input"))
    else:
        input_dir = input_path

    model, in_ch, resolved_scale = load_model(
        checkpoint_path, device=device, sr_scale=sr_scale
    )
    engine = TiledInferenceEngine(
        model, tile_size=tile_size, overlap=overlap, sr_scale=resolved_scale,
        tile_batch_size=tile_batch_size, device=device,
    )

    paths = _list_samples(input_dir)
    if len(paths) == 0:
        raise FileNotFoundError(f"No .npy/image samples found under {input_dir}")

    total_time = 0.0
    for path in paths:
        arr = load_array(path)
        tensor = _to_chw_tensor(arr)  # (C,H,W), native dtype/scale preserved

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        restored = engine.infer(tensor)

        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        total_time += dt

        stem = os.path.splitext(os.path.basename(path))[0]
        restored_np = restored.cpu().numpy()  # (C,H,W)

        # Native-format output: matches the contest's .npy tiles exactly
        # (drop the channel axis back to (H,W) for single-channel data).
        npy_out = restored_np[0] if restored_np.shape[0] == 1 else restored_np.transpose(1, 2, 0)
        np.save(os.path.join(output_dir, f"{stem}_restored.npy"), npy_out)

        if save_png_preview:
            preview = restored.clamp(0.0, 1.0).cpu().numpy()
            preview = preview[0] if preview.shape[0] == 1 else preview.transpose(1, 2, 0)
            preview_u8 = (preview * 255.0).astype(np.uint8)
            Image.fromarray(preview_u8).save(os.path.join(output_dir, f"{stem}_preview.png"))

        print(f"[{stem}] -> restored in {dt * 1000:.1f} ms")

    if paths:
        print(f"Processed {len(paths)} samples, avg {total_time / len(paths) * 1000:.1f} ms/sample")


# --------------------------------------------------------------------------- #
# ONNX / TensorRT export
# --------------------------------------------------------------------------- #

def export_to_tensorrt(
    checkpoint_path: str,
    onnx_path: str = "semicon_nafnet.onnx",
    tile_size: int = 128,
    sr_scale: int = None,
    opset: int = 17,
    dynamic_batch: bool = True,
):
    """
    Exports the trained model to ONNX with dynamic axes for batch size (and
    optionally spatial dims), suitable as input to `trtexec` /
    `torch_tensorrt` for building a TensorRT engine for ultra-fast
    deployment on the contest's evaluation GPU.

    Note: actual TensorRT engine building (trtexec --onnx=... --fp16
    --saveEngine=...) is intentionally left as an external CLI step, since
    it depends on the target GPU's installed TensorRT version at contest
    time; this function only produces the portable ONNX intermediate.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, in_ch, resolved_scale = load_model(
        checkpoint_path, device=device, sr_scale=sr_scale
    )
    model.eval()

    dummy_input = torch.randn(1, in_ch, tile_size, tile_size, device=device)

    dynamic_axes = {"input": {}, "output": {}}
    if dynamic_batch:
        dynamic_axes["input"][0] = "batch"
        dynamic_axes["output"][0] = "batch"
    # Height/width kept dynamic too, since tile size at deployment time may
    # differ slightly from the training/export default.
    dynamic_axes["input"][2] = "height"
    dynamic_axes["input"][3] = "width"
    dynamic_axes["output"][2] = "height_out"
    dynamic_axes["output"][3] = "width_out"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"ONNX model exported to {onnx_path}")
    print(
        "Next step for TensorRT deployment:\n"
        f"  trtexec --onnx={onnx_path} --fp16 --saveEngine=semicon_nafnet.trt "
        f"--minShapes=input:1x3x{tile_size//2}x{tile_size//2} "
        f"--optShapes=input:1x3x{tile_size}x{tile_size} "
        f"--maxShapes=input:4x3x{tile_size*2}x{tile_size*2}"
    )


# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    infer_p = sub.add_parser("infer")
    infer_p.add_argument("--checkpoint", type=str, required=True)
    infer_p.add_argument("--input_path", type=str, required=True,
                          help="Folder OR zip of samples, e.g. Test_NoisyLR.zip")
    infer_p.add_argument("--output_dir", type=str, required=True)
    infer_p.add_argument("--tile_size", type=int, default=128,
                          help="128 matches the observed NoisyLR sample size "
                               "exactly (single-tile, no real tiling needed); "
                               "raise this for larger production images")
    infer_p.add_argument("--overlap", type=int, default=16)
    infer_p.add_argument("--sr_scale", type=int, default=None,
                          help="Overrides the scale stored in the checkpoint; "
                               "omit to auto-use what train.py detected")
    infer_p.add_argument("--tile_batch_size", type=int, default=8)

    export_p = sub.add_parser("export")
    export_p.add_argument("--checkpoint", type=str, required=True)
    export_p.add_argument("--onnx_path", type=str, default="semicon_nafnet.onnx")
    export_p.add_argument("--tile_size", type=int, default=128)
    export_p.add_argument("--sr_scale", type=int, default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "infer":
        run_folder_inference(
            checkpoint_path=args.checkpoint,
            input_path=args.input_path,
            output_dir=args.output_dir,
            tile_size=args.tile_size,
            overlap=args.overlap,
            sr_scale=args.sr_scale,
            tile_batch_size=args.tile_batch_size,
        )
    elif args.mode == "export":
        export_to_tensorrt(
            checkpoint_path=args.checkpoint,
            onnx_path=args.onnx_path,
            tile_size=args.tile_size,
            sr_scale=args.sr_scale,
        )
