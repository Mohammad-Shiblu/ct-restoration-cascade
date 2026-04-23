#!/usr/bin/env bash
# run_baselines.sh
# ================
# Runs all baseline evaluations and model trainings sequentially.
#
# Order
# -----
#   1. BM3D evaluation          (classical baseline, ~2-3 h)
#   2. Single-stage UNet        (full-capacity, 80 epochs max)
#   3. Naive cascade — e2e      (direct Stage1, detach=false, 80 epochs max)
#   4. Naive cascade — detach   (direct Stage1, detach=true,  80 epochs max)
#
# Usage
# -----
#   nohup bash run_baselines.sh > output/baselines/run_log.txt 2>&1 &
#   echo "PID: $!"

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

mkdir -p output/baselines

echo "============================================================"
echo "  BASELINE PIPELINE — $(date)"
echo "============================================================"
echo ""

# ── 1. BM3D ──────────────────────────────────────────────────────────────────
echo "[1/4] BM3D evaluation"
echo "      Results → output/baselines/bm3d/"
echo "      Started: $(date)"
echo ""

conda run -n tensor python baselines/bm3d_eval.py

echo ""
echo "      BM3D done: $(date)"
echo ""

# ── 2. Single-stage UNet (full capacity) ─────────────────────────────────────
echo "[2/4] Single-stage UNet — full capacity [64,128,256,512], 80 epochs max"
echo "      Config  → config/lodopab_ds_baseline_unet_train.json"
echo "      Started: $(date)"
echo ""

conda run -n tensor python main_lodopab.py config/lodopab_ds_baseline_unet_train.json

echo ""
echo "      Single UNet done: $(date)"
echo ""

# ── 3. Naive cascade — end-to-end ────────────────────────────────────────────
echo "[3/4] Naive cascade — direct Stage1, detach=false, 80 epochs max"
echo "      Config  → config/lodopab_ds_naive_cascade_e2e_train.json"
echo "      Started: $(date)"
echo ""

conda run -n tensor python main_lodopab.py config/lodopab_ds_naive_cascade_e2e_train.json

echo ""
echo "      Naive cascade (e2e) done: $(date)"
echo ""

# ── 4. Naive cascade — detached ──────────────────────────────────────────────
echo "[4/4] Naive cascade — direct Stage1, detach=true, 80 epochs max"
echo "      Config  → config/lodopab_ds_naive_cascade_detach_train.json"
echo "      Started: $(date)"
echo ""

conda run -n tensor python main_lodopab.py config/lodopab_ds_naive_cascade_detach_train.json

echo ""
echo "      Naive cascade (detach) done: $(date)"
echo ""

echo "============================================================"
echo "  ALL BASELINES COMPLETE — $(date)"
echo "============================================================"
