#!/usr/bin/env bash
# =============================================================================
# setup_environment.sh
# =============================================================================
# Set up the full training environment from scratch (conda + torch_radon).
#
# Tested environments
# -------------------
#   CUDA 12.6  — PyTorch 2.x, Python 3.12 (Vast.ai / local)
#   CUDA 13.0  — PyTorch 2.11.0+cu130, Python 3.12 (Vast.ai)
#
# Usage
# -----
#   bash scripts/setup_environment.sh [ENV_NAME]
#
#   ENV_NAME  optional conda environment name  (default: ct_denoising)
#
# What this script does
# ---------------------
#   1. Create a conda environment with Python 3.12
#   2. Install PyTorch with the appropriate CUDA index URL
#   3. Install Python dependencies from requirements.txt
#   4. Compile torch_radon v1.0.0 from source (4–5 patches, CUDA-version-aware)
#   5. Verify torch_radon can run a forward projection
#   6. Build the metal-implant sinogram library  (data/metal/metal_library.npz)
#
# torch_radon patches applied
# ---------------------------
#   2a  Restrict GPU architectures to sm_80, sm_86  (all CUDA versions)
#   2b  Remove CUFFT_INCOMPLETE_PARAMETER_LIST       (CUDA 13+ only)
#   2c  Replace torch.rfft/irfft with torch.fft API  (all versions)
#   2d  Replace np.int with int                      (all versions)
#   2e  Fix Fourier-filter broadcast shape           (all versions)
# =============================================================================

set -euo pipefail

ENV_NAME="${1:-ct_denoising}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Environment : $ENV_NAME"
echo "  Repo dir    : $REPO_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Detect CUDA major version ─────────────────────────────────────────────────
# Try nvcc first (system-wide); fall back to nvidia-smi driver report.
CUDA_MAJOR=$(
    nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' ||
    nvidia-smi 2>/dev/null   | grep -oP 'CUDA Version: \K[0-9]+' ||
    echo "0"
)
echo "Detected CUDA major version: $CUDA_MAJOR"

# Map to the closest PyTorch wheel index
if   [ "$CUDA_MAJOR" -ge 13 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu130"
elif [ "$CUDA_MAJOR" -ge 12 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
elif [ "$CUDA_MAJOR" -ge 11 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
else
    TORCH_INDEX=""   # CPU-only / let conda decide
fi
echo "PyTorch wheel index: ${TORCH_INDEX:-'(default — CPU or conda)'}"
echo ""

# ── Step 1: Create conda environment ─────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[1/5] Creating conda environment '$ENV_NAME' (Python 3.12) ..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
conda create -y -n "$ENV_NAME" python=3.12
echo ""

# ── Step 2: Install PyTorch ───────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[2/5] Installing PyTorch ..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -n "$TORCH_INDEX" ]; then
    conda run -n "$ENV_NAME" pip install torch torchvision --index-url "$TORCH_INDEX"
else
    conda run -n "$ENV_NAME" conda install -y pytorch torchvision cpuonly -c pytorch
fi
echo ""

# ── Step 3: Python dependencies ──────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[3/5] Installing Python dependencies from requirements.txt ..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
conda run -n "$ENV_NAME" pip install -r requirements.txt
echo ""

# ── Step 4: Compile torch_radon with compatibility patches ────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[4/5] Compiling torch_radon v1.0.0 from source ..."
echo "      (matteo-ronchetti/torch-radon — unmaintained, requires patches)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Install cuda-nvcc into the env so setup.py can compile CUDA kernels.
# (nvcc is conda-managed and may not be present system-wide.)
echo "  Installing cuda-nvcc into $ENV_NAME ..."
conda install -y -n "$ENV_NAME" -c nvidia cuda-nvcc="$CUDA_MAJOR.*" 2>/dev/null || \
    echo "  Warning: cuda-nvcc conda install failed — compilation may fail if nvcc is not in PATH."

rm -rf /tmp/torch-radon
git clone https://github.com/matteo-ronchetti/torch-radon.git /tmp/torch-radon

# Patch 2a: CUDA 13 dropped Pascal (sm_60/70) and Turing (sm_75) support.
# Restrict to Ampere (sm_80) and RTX 30xx (sm_86) to avoid build errors on
# any CUDA version, and to keep compilation fast.
echo "  [patch 2a] Restricting GPU architectures to sm_80, sm_86 ..."
python3 - << 'PYEOF'
f = '/tmp/torch-radon/build_tools/__init__.py'
txt = open(f).read()
# NOTE: the upstream source has a typo — "compute_capabilites" (missing 'i').
txt = txt.replace(
    'def build(compute_capabilites=(60, 70, 75, 80, 86),',
    'def build(compute_capabilites=(80, 86),'
)
open(f, 'w').write(txt)
print('  done.')
PYEOF

# Patch 2b: CUFFT_INCOMPLETE_PARAMETER_LIST was removed from cufft.h in
# CUDA 13. Comment out the switch-case branch that references it.
if [ "$CUDA_MAJOR" -ge 13 ]; then
    echo "  [patch 2b] Removing CUFFT_INCOMPLETE_PARAMETER_LIST (CUDA 13+) ..."
    python3 - << 'PYEOF'
f = '/tmp/torch-radon/include/utils.h'
txt = open(f).read()
txt = txt.replace(
    'case CUFFT_INCOMPLETE_PARAMETER_LIST:',
    '// case CUFFT_INCOMPLETE_PARAMETER_LIST: // removed in CUDA 13'
)
open(f, 'w').write(txt)
print('  done.')
PYEOF
else
    echo "  [patch 2b] Skipped — CUFFT_INCOMPLETE_PARAMETER_LIST present in CUDA $CUDA_MAJOR"
fi

echo "  Compiling CUDA kernels (this takes 2–5 minutes) ..."
cd /tmp/torch-radon && conda run -n "$ENV_NAME" python setup.py install
cd "$REPO_DIR"

# Patches 2c/2d/2e operate on the installed package, so locate it dynamically.
PKG_DIR=$(conda run -n "$ENV_NAME" python3 -c \
    "import torch_radon, os; print(os.path.dirname(torch_radon.__file__))")
echo "  torch_radon installed at: $PKG_DIR"

# Patch 2c: torch.rfft and torch.irfft were removed in PyTorch 2.0.
# Replace with the modern torch.fft.fft / torch.fft.ifft equivalents.
echo "  [patch 2c] Replacing torch.rfft/irfft with torch.fft API ..."
python3 - << PYEOF
f = '$PKG_DIR/__init__.py'
txt = open(f).read()
txt = txt.replace(
    'sino_fft = torch.rfft(padded_sinogram, 1, normalized=True, onesided=False)',
    'sino_fft = torch.fft.fft(padded_sinogram)'
)
txt = txt.replace(
    'filtered_sinogram = torch.irfft(filtered_sino_fft, 1, normalized=True, onesided=False)',
    'filtered_sinogram = torch.real(torch.fft.ifft(filtered_sino_fft))'
)
open(f, 'w').write(txt)
print('  done.')
PYEOF

# Patch 2d: np.int was deprecated in NumPy 1.20 and removed in 1.24.
echo "  [patch 2d] Replacing np.int with int ..."
python3 - << PYEOF
f = '$PKG_DIR/filtering.py'
txt = open(f).read()
txt = txt.replace('dtype=np.int)', 'dtype=int)')
open(f, 'w').write(txt)
print('  done.')
PYEOF

# Patch 2e: torch.fft.fft returns a complex tensor of shape (B, A, N).
# The filter f has shape (1, N, 1); reshape to (1, 1, N) for correct
# broadcasting against the angle dimension.
echo "  [patch 2e] Fixing Fourier-filter broadcast shape for complex tensors ..."
python3 - << PYEOF
f = '$PKG_DIR/__init__.py'
txt = open(f).read()
txt = txt.replace(
    'filtered_sino_fft = sino_fft * f',
    'filtered_sino_fft = sino_fft * f.squeeze(2).unsqueeze(1)'
)
open(f, 'w').write(txt)
print('  done.')
PYEOF

echo ""

# ── Step 5: Verify ────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[5/5] Verifying torch_radon ..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
conda run -n "$ENV_NAME" python3 - << 'PYEOF'
from torch_radon import Radon
import torch, numpy as np
angles = np.linspace(0, 3.14159, 100, endpoint=False).astype('float32')
r = Radon(64, angles, det_count=64)
x = torch.zeros(1, 64, 64).cuda()
s = r.forward(x)
print('  torch_radon OK — forward projection shape:', tuple(s.shape))
PYEOF
echo ""

# ── Done ──────────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup complete!  Activate with:  conda activate $ENV_NAME"
echo ""
echo "  Next steps:"
echo "  1. Download the LoDoPaB-CT dataset (~114 GB):"
echo "       python data_prep/download_lodopab.py"
echo ""
echo "  2. Build the metal-implant sinogram library:"
echo "       python scripts/build_metal_library.py --n 200 --device cuda"
echo ""
echo "  3. Edit main.py to select an experiment (set the CONFIG variable),"
echo "     then start training:"
echo "       python main.py"
echo ""
echo "  4. For multi-GPU runs, launch one process per GPU:"
echo "       CUDA_VISIBLE_DEVICES=0 python main.py > output/run_gpu0.log 2>&1 &"
echo "       CUDA_VISIBLE_DEVICES=1 python main.py > output/run_gpu1.log 2>&1 &"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
