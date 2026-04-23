#!/usr/bin/env bash
# run_all.sh
# ==========
# Full experimental pipeline for the LoDoPaB-CT artifact study.
#
# Run one copy of this script per GPU, passing CUDA_VISIBLE_DEVICES.
# Assign jobs across your three rented GPUs:
#
#   GPU 0 — RTX 5090  (fastest):
#       CUDA_VISIBLE_DEVICES=0 bash run_all.sh gpu0
#
#   GPU 1 — RTX 5090:
#       CUDA_VISIBLE_DEVICES=1 bash run_all.sh gpu1
#
#   GPU 2 — RTX 5060 Ti  (lighter jobs):
#       CUDA_VISIBLE_DEVICES=2 bash run_all.sh gpu2
#
# Logs are written to output/run_log_<gpu>.txt
# Use: nohup CUDA_VISIBLE_DEVICES=X bash run_all.sh gpuX > output/run_log_gpuX.txt 2>&1 &

set -euo pipefail

GPU="${1:-gpu0}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

mkdir -p output/baselines

PY="conda run -n tensor python"

echo "============================================================"
echo "  EXPERIMENT PIPELINE — GPU: $GPU — $(date)"
echo "============================================================"

case "$GPU" in

  # ── GPU 0: RTX 5090 ─────────────────────────────────────────────────────────
  gpu0)
    echo "[GPU0] Job 1/2: BM3D (CPU-bound, ~3h)"
    $PY baselines/bm3d_eval.py
    echo "[GPU0] BM3D done: $(date)"

    echo "[GPU0] Job 2/2: Single-stage U-Net [64,128,256,512] (~6h)"
    $PY main_lodopab.py config/lodopab_ds_baseline_unet_train.json
    echo "[GPU0] U-Net done: $(date)"
    ;;

  # ── GPU 1: RTX 5090 ─────────────────────────────────────────────────────────
  gpu1)
    echo "[GPU1] Job 1/2: 2-stage cascade, e2e, direct (naive) (~8h)"
    $PY main_lodopab.py config/lodopab_ds_naive_cascade_e2e_train.json
    echo "[GPU1] Cascade e2e done: $(date)"

    echo "[GPU1] Job 2/2: 2-stage cascade, e2e, residual (~8h)"
    $PY main_lodopab.py config/lodopab_ds_residual_cascade_e2e_train.json
    echo "[GPU1] Residual cascade done: $(date)"
    ;;

  # ── GPU 2: RTX 5060 Ti ──────────────────────────────────────────────────────
  gpu2)
    echo "[GPU2] Job 1/2: RED-CNN (~5h)"
    $PY main_lodopab.py config/lodopab_ds_baseline_redcnn_train.json
    echo "[GPU2] RED-CNN done: $(date)"

    echo "[GPU2] Job 2/2: 2-stage cascade, detach, direct (independent) (~8h)"
    $PY main_lodopab.py config/lodopab_ds_naive_cascade_detach_train.json
    echo "[GPU2] Cascade detach done: $(date)"
    ;;

  *)
    echo "Unknown GPU label '$GPU'. Use: gpu0, gpu1, or gpu2"
    exit 1
    ;;
esac

echo "============================================================"
echo "  ALL JOBS DONE — $GPU — $(date)"
echo "============================================================"
