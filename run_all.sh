#!/usr/bin/env bash
# run_all.sh
# ==========
# Full experimental pipeline — 4× RTX 3090, 7 training jobs.
# BM3D is excluded (already evaluated locally).
#
# Launch all four GPU workers simultaneously from one terminal:
#
#   mkdir -p output
#   CUDA_VISIBLE_DEVICES=0 bash run_all.sh gpu0 > output/run_log_gpu0.txt 2>&1 &
#   CUDA_VISIBLE_DEVICES=1 bash run_all.sh gpu1 > output/run_log_gpu1.txt 2>&1 &
#   CUDA_VISIBLE_DEVICES=2 bash run_all.sh gpu2 > output/run_log_gpu2.txt 2>&1 &
#   CUDA_VISIBLE_DEVICES=3 bash run_all.sh gpu3 > output/run_log_gpu3.txt 2>&1 &
#
# Monitor any GPU:
#   tail -f output/run_log_gpu0.txt
#
# Job assignment (estimated finish time from a cold start):
#   GPU 0 (~16 h): dual-domain residual detach (~10h) → single UNet (~6h)
#   GPU 1 (~14 h): naive cascade e2e (~8h)  → RED-CNN (~5h)
#   GPU 2 (~16 h): naive cascade detach (~8h) → residual cascade e2e (~8h)
#   GPU 3 (~ 8 h): residual cascade detach (~8h)

set -euo pipefail

GPU="${1:-gpu0}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

mkdir -p output checkpoints

PY="python"

echo "============================================================"
echo "  EXPERIMENT PIPELINE — GPU: $GPU — $(date)"
echo "============================================================"

run_job() {
    local label="$1"
    local config="$2"
    echo ""
    echo ">>> [$GPU] START: $label — $(date)"
    $PY main_lodopab.py "$config"
    echo ">>> [$GPU] DONE:  $label — $(date)"
    echo ""
}

case "$GPU" in

  # ── GPU 0 (~16 h) ────────────────────────────────────────────────────────────
  # Longest job first so the GPU is never idle waiting for a short warm-up job.
  gpu0)
    run_job "Residual cascade — detach — dual-domain" \
            config/lodopab_ds_residual_cascade_detach_dual_train.json

    run_job "Single-stage U-Net [64,128,256,512]" \
            config/lodopab_ds_baseline_unet_train.json
    ;;

  # ── GPU 1 (~14 h) ────────────────────────────────────────────────────────────
  gpu1)
    run_job "Naive cascade — e2e (joint, no detach)" \
            config/lodopab_ds_naive_cascade_e2e_train.json

    run_job "RED-CNN baseline" \
            config/lodopab_ds_baseline_redcnn_train.json
    ;;

  # ── GPU 2 (~16 h) ────────────────────────────────────────────────────────────
  gpu2)
    run_job "Naive cascade — detach (independent stages)" \
            config/lodopab_ds_naive_cascade_detach_train.json

    run_job "Residual cascade — e2e (joint, no detach)" \
            config/lodopab_ds_residual_cascade_e2e_train.json
    ;;

  # ── GPU 3 (~8 h) ─────────────────────────────────────────────────────────────
  gpu3)
    run_job "Residual cascade — detach (independent stages)" \
            config/lodopab_ds_residual_cascade_detach_train.json
    ;;

  *)
    echo "Unknown GPU label '$GPU'. Use: gpu0, gpu1, gpu2, or gpu3"
    exit 1
    ;;
esac

echo "============================================================"
echo "  ALL JOBS DONE — $GPU — $(date)"
echo "============================================================"
