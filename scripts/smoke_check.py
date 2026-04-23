"""
smoke_test.py — Quick end-to-end sanity check for the full training pipeline.

Uses test_local=True which limits data to:
  train: 32 samples | val: 8 samples | test: 8 samples

Each config runs for at most 2 epochs so the full test finishes in
~5-10 minutes on a GPU (no GPU → falls back to CPU, slower).

Usage (from repo root):
    python scripts/smoke_test.py                      # all configs
    python scripts/smoke_test.py --config lodopab_ds_baseline_unet_train.json
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch

# resolve repo root regardless of cwd
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

os.environ.setdefault("TORCH_HOME", str(REPO / "torch_cache"))

from sinogram_trainer.lodopab_ds_trainer import LoDoPaBDeepSupervisionTrainer
from utils.help import setup_logger

CONFIGS = [
    "config/lodopab_ds_baseline_unet_train.json",
    "config/lodopab_ds_baseline_redcnn_train.json",
    "config/lodopab_ds_naive_cascade_e2e_train.json",
    "config/lodopab_ds_naive_cascade_detach_train.json",
    "config/lodopab_ds_residual_cascade_e2e_train.json",
]

SMOKE_OVERRIDES = {
    "epochs": 2,
    "patience": 999,
    "batch_size": 4,
    "num_workers": 2,
}


def run_one(config_path: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  SMOKE TEST: {config_path}")
    print(f"{'='*60}")
    try:
        with open(config_path) as f:
            config = json.load(f)

        config.update(SMOKE_OVERRIDES)
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        config["checkpoints_dir"] = "checkpoints_smoke"
        config["output_dir"] = "output_smoke"

        logger = setup_logger(config)
        trainer = LoDoPaBDeepSupervisionTrainer(
            config=config, logger=logger, test_local=True
        )
        trainer.run()
        print(f"  PASSED: {config_path}")
        return True
    except Exception:
        print(f"  FAILED: {config_path}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Single config file to test")
    args = parser.parse_args()

    configs = [args.config] if args.config else CONFIGS

    results = {}
    for cfg in configs:
        results[cfg] = run_one(cfg)

    print(f"\n{'='*60}")
    print("  SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    all_passed = True
    for cfg, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {cfg}")
        if not passed:
            all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
