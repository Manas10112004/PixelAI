<#
.SYNOPSIS
    SEMICON 2026 HPC restoration pipeline, end to end -- PowerShell version.
    Equivalent to run.sh, for people running plain Windows PowerShell / the
    VS Code integrated terminal without WSL or Git Bash available.

.EXAMPLE
    .\run.ps1 -TrainZip train.zip -TestZip Test_NoisyLR.zip

.EXAMPLE
    # Just re-run inference against an existing checkpoint:
    .\run.ps1 -TrainZip train.zip -TestZip Test_NoisyLR.zip -SkipTrain

.EXAMPLE
    # Also export ONNX for TensorRT:
    .\run.ps1 -TrainZip train.zip -TestZip Test_NoisyLR.zip -Export
#>

param(
    [string]$TrainZip = "train.zip",
    [string]$TestZip = "Test_NoisyLR.zip",
    [string]$WorkDir = "./data_ingested",
    [string]$CheckpointDir = "./checkpoints",
    [string]$OutputDir = "./restored_output",
    [int]$Epochs = 100,
    [int]$BatchSize = 16,
    [double]$Lr = 2e-4,
    [int]$NumWorkers = 4,
    [int]$CropSize = 128,
    [double]$ValSplit = 0.1,
    [string]$CompileMode = "max-autotune",
    [int]$TileSize = 128,
    [int]$Overlap = 16,
    [int]$TileBatchSize = 8,
    [switch]$NoInstall,
    [switch]$SkipTrain,
    [switch]$SkipInfer,
    [switch]$Export
)

$ErrorActionPreference = "Stop"

function Log($msg) {
    Write-Host ""
    Write-Host "[run.ps1] $msg" -ForegroundColor Cyan
}

# Always run from the script's own directory, same as run.sh's cd "$SCRIPT_DIR".
Set-Location -Path $PSScriptRoot

# --------------------------------------------------------------------------- #
# 0. Sanity checks
# --------------------------------------------------------------------------- #
if (-not (Test-Path $TrainZip)) {
    Write-Error "Train zip not found at '$TrainZip'. Pass -TrainZip <path> pointing at your actual file."
    exit 1
}
if (-not (Test-Path $TestZip)) {
    Write-Error "Test zip not found at '$TestZip'. Pass -TestZip <path> pointing at your actual file."
    exit 1
}

$hasGpu = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($hasGpu) {
    Log "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} else {
    Log "WARNING: nvidia-smi not found -- this will fall back to CPU (slow, and torch.compile/AMP are effectively disabled)."
}

# Figure out which python to use: prefer 'python', fall back to 'py'.
$pythonCmd = "python"
if (-not (Get-Command $pythonCmd -ErrorAction SilentlyContinue)) {
    $pythonCmd = "py"
    if (-not (Get-Command $pythonCmd -ErrorAction SilentlyContinue)) {
        Write-Error "Neither 'python' nor 'py' was found on PATH."
        exit 1
    }
}

# --------------------------------------------------------------------------- #
# 1. Dependencies
# --------------------------------------------------------------------------- #
if (-not $NoInstall) {
    Log "Installing requirements.txt ..."
    & $pythonCmd -m pip install -r requirements.txt -q
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pip install failed (exit code $LASTEXITCODE)."
        exit $LASTEXITCODE
    }
} else {
    Log "Skipping dependency install (-NoInstall)"
}

# --------------------------------------------------------------------------- #
# 2. Data ingestion preview
# --------------------------------------------------------------------------- #
Log "Ingesting $TrainZip + $TestZip -> $WorkDir ..."
& $pythonCmd data_ingestion.py `
    --train_zip $TrainZip `
    --test_zip $TestZip `
    --work_dir $WorkDir `
    --crop_size $CropSize
if ($LASTEXITCODE -ne 0) {
    Write-Error "data_ingestion.py failed (exit code $LASTEXITCODE). Fix the reported issue before continuing."
    exit $LASTEXITCODE
}

# --------------------------------------------------------------------------- #
# 3. Train
# --------------------------------------------------------------------------- #
if (-not $SkipTrain) {
    Log "Training (epochs=$Epochs batch_size=$BatchSize lr=$Lr compile_mode=$CompileMode) ..."
    & $pythonCmd train.py `
        --train_zip $TrainZip `
        --test_zip $TestZip `
        --work_dir $WorkDir `
        --checkpoint_dir $CheckpointDir `
        --epochs $Epochs `
        --batch_size $BatchSize `
        --lr $Lr `
        --num_workers $NumWorkers `
        --crop_size $CropSize `
        --val_split $ValSplit `
        --compile_mode $CompileMode
    if ($LASTEXITCODE -ne 0) {
        Write-Error "train.py failed (exit code $LASTEXITCODE)."
        exit $LASTEXITCODE
    }
} else {
    Log "Skipping training (-SkipTrain)"
}

$bestCkpt = Join-Path $CheckpointDir "best_model.pt"
if (-not (Test-Path $bestCkpt)) {
    Write-Error "Expected checkpoint not found at $bestCkpt"
    exit 1
}

# --------------------------------------------------------------------------- #
# 4. Tiled inference over Test_NoisyLR.zip
# --------------------------------------------------------------------------- #
if (-not $SkipInfer) {
    Log "Running tiled inference on $TestZip -> $OutputDir ..."
    & $pythonCmd eval_hpc.py infer `
        --checkpoint $bestCkpt `
        --input_path $TestZip `
        --output_dir $OutputDir `
        --tile_size $TileSize `
        --overlap $Overlap `
        --tile_batch_size $TileBatchSize
    if ($LASTEXITCODE -ne 0) {
        Write-Error "eval_hpc.py infer failed (exit code $LASTEXITCODE)."
        exit $LASTEXITCODE
    }
} else {
    Log "Skipping inference (-SkipInfer)"
}

# --------------------------------------------------------------------------- #
# 5. Optional ONNX export
# --------------------------------------------------------------------------- #
if ($Export) {
    Log "Exporting ONNX for TensorRT deployment ..."
    & $pythonCmd eval_hpc.py export `
        --checkpoint $bestCkpt `
        --onnx_path (Join-Path $CheckpointDir "semicon_nafnet.onnx") `
        --tile_size $TileSize
    if ($LASTEXITCODE -ne 0) {
        Write-Error "eval_hpc.py export failed (exit code $LASTEXITCODE)."
        exit $LASTEXITCODE
    }
}

Log "Done. Checkpoint: $bestCkpt | Restored output: $OutputDir"
