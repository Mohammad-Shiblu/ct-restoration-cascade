#!/usr/bin/env bash
# setup_vastai_cuda13.sh
# =====================
# Full setup for a fresh Vast.ai instance with CUDA 13.0 / PyTorch 2.11.
# Run from /workspace after cloning the repo.
#
# Usage:
#   cd /workspace/Masters-Thesis
#   bash scripts/setup_vastai_cuda13.sh

set -euo pipefail

echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt

echo "[2/4] Compiling torch_radon..."
git clone https://github.com/matteo-ronchetti/torch-radon.git /tmp/torch-radon

# Fix 1: remove GPU architectures unsupported by CUDA 13
python - << 'PYEOF'
f = '/tmp/torch-radon/build_tools/__init__.py'
txt = open(f).read()
txt = txt.replace(
    'def build(compute_capabilities=(60, 70, 75, 80, 86),',
    'def build(compute_capabilities=(80, 86),'
)
open(f, 'w').write(txt)
print('  build_tools: architecture fix applied')
PYEOF

# Fix 2: CUFFT_INCOMPLETE_PARAMETER_LIST removed in CUDA 13
python - << 'PYEOF'
f = '/tmp/torch-radon/include/utils.h'
txt = open(f).read()
txt = txt.replace(
    'case CUFFT_INCOMPLETE_PARAMETER_LIST:',
    '// case CUFFT_INCOMPLETE_PARAMETER_LIST: // removed in CUDA 13'
)
open(f, 'w').write(txt)
print('  utils.h: CUFFT fix applied')
PYEOF

cd /tmp/torch-radon && python setup.py install
cd /workspace/Masters-Thesis

# Fix 3: torch.rfft removed in PyTorch 2.0
python - << 'PYEOF'
f = '/venv/main/lib/python3.12/site-packages/torch_radon/__init__.py'
txt = open(f).read()
txt = txt.replace(
    'sino_fft = torch.rfft(padded_sinogram, 1, normalized=True, onesided=False)',
    'sino_fft = torch.fft.fft(padded_sinogram)'
)
txt = txt.replace(
    'filtered_sinogram = torch.irfft(filtered_sino_fft, 1, normalized=True, onesided=False)',
    'filtered_sinogram = torch.real(torch.fft.ifft(filtered_sino_fft))'
)
txt = txt.replace(
    'filtered_sino_fft = sino_fft * f',
    'filtered_sino_fft = sino_fft * f.squeeze(2).unsqueeze(1)'
)
open(f, 'w').write(txt)
print('  __init__.py: torch.fft fix applied')
PYEOF

# Fix 4: np.int removed in NumPy 1.24+
python - << 'PYEOF'
f = '/venv/main/lib/python3.12/site-packages/torch_radon/filtering.py'
txt = open(f).read()
txt = txt.replace('dtype=np.int)', 'dtype=int)')
open(f, 'w').write(txt)
print('  filtering.py: np.int fix applied')
PYEOF

echo "[3/4] Verifying torch_radon..."
python -c "from torch_radon import Radon; print('  torch_radon OK')"

echo "[4/4] Building metal library..."
mkdir -p data/metal
python scripts/build_metal_library.py --n 200 --device cuda

echo ""
echo "Setup complete. Start training once dataset download finishes:"
echo "  tail -f /workspace/download.log"
echo "  bash scripts/start_training.sh"
