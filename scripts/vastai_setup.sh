#!/usr/bin/env bash
# vastai_setup.sh
# ===============
# Run this ONCE after SSH-ing into a fresh Vast.ai instance to:
#   1. Install Miniconda
#   2. Create the `tensor` conda environment
#   3. Install all Python dependencies
#   4. Clone the repo
#   5. Start the LoDoPaB dataset download (background)
#
# Usage (paste into the Vast.ai terminal or run via SSH):
#   bash vastai_setup.sh
#
# After this script, start training with:
#   cd /workspace/Masters-Thesis
#   tmux new -s train
#   CUDA_VISIBLE_DEVICES=0 bash run_all.sh gpu0

set -euo pipefail

REPO_URL="https://github.com/YOUR_GITHUB_USERNAME/Masters-Thesis.git"  # <-- change this
WORKSPACE="/workspace"
CONDA_DIR="$WORKSPACE/miniconda3"
ENV_NAME="tensor"

# ── 1. Install Miniconda if not present ──────────────────────────────────────
if [ ! -f "$CONDA_DIR/bin/conda" ]; then
    echo "[1/5] Installing Miniconda …"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
    bash /tmp/mc.sh -b -p "$CONDA_DIR"
    rm /tmp/mc.sh
fi

export PATH="$CONDA_DIR/bin:$PATH"
conda init bash
# shellcheck disable=SC1090
source "$CONDA_DIR/etc/profile.d/conda.sh"

# ── 2. Create environment ─────────────────────────────────────────────────────
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "[2/5] Creating conda env '$ENV_NAME' …"
    conda create -y -n "$ENV_NAME" python=3.10
fi

conda activate "$ENV_NAME"

# ── 3. Install dependencies ───────────────────────────────────────────────────
echo "[3/5] Installing Python packages …"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy h5py tqdm matplotlib scikit-image requests bm3d \
            astra-toolbox pyqtgraph

# If you use numba / diffCT:
# conda install -y -c numba numba cudatoolkit

# ── 4. Clone repo ─────────────────────────────────────────────────────────────
echo "[4/5] Cloning repository …"
cd "$WORKSPACE"
if [ ! -d "Masters-Thesis" ]; then
    git clone "$REPO_URL" Masters-Thesis
fi
cd Masters-Thesis

# ── 5. Download dataset (background, takes ~2-3 h) ───────────────────────────
echo "[5/5] Starting LoDoPaB dataset download in background …"
export LODOPAB_DATA_PATH="$WORKSPACE/lodopab_data"
mkdir -p "$LODOPAB_DATA_PATH"
nohup conda run -n "$ENV_NAME" python data_prep/download_lodopab.py \
    > "$WORKSPACE/download.log" 2>&1 &
echo "  Download PID: $!"
echo "  Monitor with: tail -f $WORKSPACE/download.log"

echo ""
echo "Setup complete. Wait for download to finish, then:"
echo "  export LODOPAB_DATA_PATH=$LODOPAB_DATA_PATH"
echo "  cd $WORKSPACE/Masters-Thesis"
echo "  tmux new -s train"
echo "  CUDA_VISIBLE_DEVICES=0 bash run_all.sh gpu0"
