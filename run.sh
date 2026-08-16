#!/usr/bin/env bash
# =============================================================================
# run.sh -- SEMICON 2026 HPC restoration pipeline, end to end.
#
# Pipeline:
#   1. (optional) install requirements.txt
#   2. ingest Train.zip + Test_NoisyLR.zip (extraction + layout auto-detect,
#      via data_ingestion.py -- also runnable standalone to just inspect
#      what was detected)
#   3. train.py: AMP + torch.compile training run, best checkpoint saved to
#      $CHECKPOINT_DIR/best_model.pt
#   4. eval_hpc.py infer: tiled, Gaussian-blended inference over
#      Test_NoisyLR.zip, restored .npy (+ PNG preview) written to
#      $OUTPUT_DIR
#   5. (optional, --export) eval_hpc.py export: ONNX export for TensorRT
#
# Usage:
#   ./run.sh --train_zip Train.zip --test_zip Test_NoisyLR.zip
#
# All steps are individually skippable/resumable -- e.g. re-run with
# --skip_train to just re-run inference against an existing checkpoint.
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# Defaults (override via flags below)
# --------------------------------------------------------------------------- #
TRAIN_ZIP="Train.zip"
TEST_ZIP="Test_NoisyLR.zip"
WORK_DIR="./data_ingested"
CHECKPOINT_DIR="./checkpoints"
OUTPUT_DIR="./restored_output"
EPOCHS=100
BATCH_SIZE=16
LR=2e-4
NUM_WORKERS=4
CROP_SIZE=128
VAL_SPLIT=0.1
COMPILE_MODE="max-autotune"
TILE_SIZE=128
OVERLAP=16
TILE_BATCH_SIZE=8

INSTALL_DEPS=1
SKIP_TRAIN=0
SKIP_INFER=0
DO_EXPORT=0

# --------------------------------------------------------------------------- #
# Arg parsing
# --------------------------------------------------------------------------- #
while [[ $# -gt 0 ]]; do
  case "$1" in
    --train_zip) TRAIN_ZIP="$2"; shift 2 ;;
    --test_zip) TEST_ZIP="$2"; shift 2 ;;
    --work_dir) WORK_DIR="$2"; shift 2 ;;
    --checkpoint_dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --crop_size) CROP_SIZE="$2"; shift 2 ;;
    --val_split) VAL_SPLIT="$2"; shift 2 ;;
    --compile_mode) COMPILE_MODE="$2"; shift 2 ;;
    --tile_size) TILE_SIZE="$2"; shift 2 ;;
    --overlap) OVERLAP="$2"; shift 2 ;;
    --tile_batch_size) TILE_BATCH_SIZE="$2"; shift 2 ;;
    --no_install) INSTALL_DEPS=0; shift ;;
    --skip_train) SKIP_TRAIN=1; shift ;;
    --skip_infer) SKIP_INFER=1; shift ;;
    --export) DO_EXPORT=1; shift ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log() { echo -e "\n\033[1;36m[run.sh]\033[0m $*"; }

# --------------------------------------------------------------------------- #
# 0. Sanity checks
# --------------------------------------------------------------------------- #
if [[ ! -f "$TRAIN_ZIP" ]]; then
  echo "ERROR: --train_zip not found at '$TRAIN_ZIP'." >&2
  echo "       Pass the actual path, e.g. ./run.sh --train_zip /path/Train.zip --test_zip /path/Test_NoisyLR.zip" >&2
  exit 1
fi
if [[ ! -f "$TEST_ZIP" ]]; then
  echo "ERROR: --test_zip not found at '$TEST_ZIP'." >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  log "GPU detected:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  log "WARNING: nvidia-smi not found -- this will fall back to CPU (slow, and torch.compile/AMP paths are effectively disabled)."
fi

# --------------------------------------------------------------------------- #
# 1. Dependencies
# --------------------------------------------------------------------------- #
if [[ "$INSTALL_DEPS" -eq 1 ]]; then
  log "Installing requirements.txt ..."
  pip install -r requirements.txt --break-system-packages -q || pip install -r requirements.txt -q
else
  log "Skipping dependency install (--no_install)"
fi

# --------------------------------------------------------------------------- #
# 2. Data ingestion preview (extraction happens for real inside train.py /
#    eval_hpc.py too, but running it here first surfaces layout-detection
#    problems -- e.g. an unrecognized Train.zip structure -- before we spend
#    time compiling the model).
# --------------------------------------------------------------------------- #
log "Ingesting $TRAIN_ZIP + $TEST_ZIP -> $WORK_DIR ..."
python3 data_ingestion.py \
  --train_zip "$TRAIN_ZIP" \
  --test_zip "$TEST_ZIP" \
  --work_dir "$WORK_DIR" \
  --crop_size "$CROP_SIZE"

# --------------------------------------------------------------------------- #
# 3. Train
# --------------------------------------------------------------------------- #
if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  log "Training (epochs=$EPOCHS batch_size=$BATCH_SIZE lr=$LR compile_mode=$COMPILE_MODE) ..."
  python3 train.py \
    --train_zip "$TRAIN_ZIP" \
    --test_zip "$TEST_ZIP" \
    --work_dir "$WORK_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --num_workers "$NUM_WORKERS" \
    --crop_size "$CROP_SIZE" \
    --val_split "$VAL_SPLIT" \
    --compile_mode "$COMPILE_MODE"
else
  log "Skipping training (--skip_train)"
fi

BEST_CKPT="$CHECKPOINT_DIR/best_model.pt"
if [[ ! -f "$BEST_CKPT" ]]; then
  echo "ERROR: expected checkpoint not found at $BEST_CKPT" >&2
  exit 1
fi

# --------------------------------------------------------------------------- #
# 4. Tiled inference over Test_NoisyLR.zip
# --------------------------------------------------------------------------- #
if [[ "$SKIP_INFER" -eq 0 ]]; then
  log "Running tiled inference on $TEST_ZIP -> $OUTPUT_DIR ..."
  python3 eval_hpc.py infer \
    --checkpoint "$BEST_CKPT" \
    --input_path "$TEST_ZIP" \
    --output_dir "$OUTPUT_DIR" \
    --tile_size "$TILE_SIZE" \
    --overlap "$OVERLAP" \
    --tile_batch_size "$TILE_BATCH_SIZE"
else
  log "Skipping inference (--skip_infer)"
fi

# --------------------------------------------------------------------------- #
# 5. Optional ONNX export
# --------------------------------------------------------------------------- #
if [[ "$DO_EXPORT" -eq 1 ]]; then
  log "Exporting ONNX for TensorRT deployment ..."
  python3 eval_hpc.py export \
    --checkpoint "$BEST_CKPT" \
    --onnx_path "$CHECKPOINT_DIR/semicon_nafnet.onnx" \
    --tile_size "$TILE_SIZE"
fi

log "Done. Checkpoint: $BEST_CKPT | Restored output: $OUTPUT_DIR"
