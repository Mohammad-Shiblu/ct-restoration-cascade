"""
Compute per-artifact-count metric breakdown and per-slice standard deviations
for the two strongest models: U-Net (large) and residual dual-filter cascade.

Outputs JSON to output/eval_per_artifact_count.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.lodopab_dataset import LoDoPaBDataset, DEFAULT_ARTIFACT_CONFIG
from utils.metrics import compute_psnr, compute_ssim, compute_rmse
from utils.model_inference import load_model
from trainer.parallel_fbp import DifferentiableParallelFBP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACT_SEED = 0
DR = 1.0


def count_artifacts_for_sample(idx: int, artifact_seed: int = 0) -> int:
    """Replay the artifact RNG for test sample `idx` and count how many fired."""
    rng_state = np.random.get_state()
    np.random.seed(artifact_seed + idx)

    cfg = DEFAULT_ARTIFACT_CONFIG
    count = 0
    # Motion
    if cfg.get("motion_blur", False):
        fired = np.random.rand() < cfg.get("motion_blur_prob", 0.5)
        if fired:
            count += 1
            # consume the 3 parameter draws
            np.random.uniform(*cfg["motion_amplitude"])
            np.random.uniform(*cfg["motion_cycles"])
            np.random.uniform(0, 2 * np.pi)
    # Ring
    if cfg.get("ring", False):
        fired = np.random.rand() < cfg.get("ring_prob", 0.5)
        if fired:
            count += 1
            n_lo, n_hi = cfg["ring_n_cols"]
            n_cols = np.random.randint(n_lo, n_hi + 1)
            # consume n_cols gain draws + n_cols flip draws
            for _ in range(n_cols):
                np.random.uniform(*cfg.get("ring_gain", (0.5, 0.95)))
                np.random.rand()  # flip with prob 0.5
    # Metal
    if cfg.get("metal", False):
        fired = np.random.rand() < cfg.get("metal_prob", 0.3)
        if fired:
            count += 1

    np.random.set_state(rng_state)
    return count


def evaluate_model(stages, fbp, ds, config):
    """Return per-slice metrics dict.

    Returns dict with keys 'psnr', 'ssim', 'rmse' each being a list of floats.
    """
    psnr_list, ssim_list, rmse_list = [], [], []
    num_stages = config["num_stages"]
    stage1_mode = config.get("stage1_mode", "residual")
    dual_domain = config.get("dual_domain", False)

    with torch.no_grad():
        for idx in range(len(ds)):
            sino_t, gt_t = ds[idx]
            sino_t = sino_t.unsqueeze(0).to(DEVICE)
            gt_t_cu = gt_t.unsqueeze(0).to(DEVICE)

            noisy_t = fbp(sino_t).clamp(0, 1)

            hann_t = sirt_t = None
            if dual_domain and num_stages > 1:
                hann_t = fbp.hann_fbp(sino_t)
                sirt_t = fbp.shepp_logan_fbp(sino_t)

            x = stages[0](noisy_t)
            for s in range(1, num_stages):
                if stage1_mode == "direct":
                    x = stages[s](x)
                else:
                    inp = torch.cat([x, hann_t, sirt_t], dim=1) if hann_t is not None else x
                    x = torch.clamp(x + stages[s](inp), 0, 1)

            pred = x.cpu()
            gt_cpu = gt_t.unsqueeze(0)
            psnr_list.append(compute_psnr(pred, gt_cpu, data_range=DR))
            ssim_list.append(compute_ssim(pred, gt_cpu, data_range=DR))
            rmse_list.append(float(compute_rmse(pred, gt_cpu).item()
                                   if hasattr(compute_rmse(pred, gt_cpu), 'item')
                                   else compute_rmse(pred, gt_cpu)))

            if (idx + 1) % 500 == 0:
                print(f"  {idx+1}/{len(ds)}")

    return {"psnr": psnr_list, "ssim": ssim_list, "rmse": rmse_list}


def summarise(metrics, artifact_counts):
    """Return mean/std overall and per artifact count 0..3."""
    psnr = np.array(metrics["psnr"])
    ssim = np.array(metrics["ssim"])
    rmse = np.array(metrics["rmse"])
    counts = np.array(artifact_counts)

    result = {
        "overall": {
            "n": len(psnr),
            "psnr_mean": float(np.mean(psnr)),
            "psnr_std":  float(np.std(psnr)),
            "ssim_mean": float(np.mean(ssim)),
            "ssim_std":  float(np.std(ssim)),
            "rmse_mean": float(np.mean(rmse)),
            "rmse_std":  float(np.std(rmse)),
        }
    }
    for k in range(4):
        mask = counts == k
        n = int(mask.sum())
        if n == 0:
            result[f"artifact_{k}"] = {"n": 0}
            continue
        result[f"artifact_{k}"] = {
            "n":         n,
            "psnr_mean": float(np.mean(psnr[mask])),
            "psnr_std":  float(np.std(psnr[mask])),
            "ssim_mean": float(np.mean(ssim[mask])),
            "ssim_std":  float(np.std(ssim[mask])),
            "rmse_mean": float(np.mean(rmse[mask])),
            "rmse_std":  float(np.std(rmse[mask])),
        }
    return result


def main():
    # ── Build artifact counts once ───────────────────────────────────────────
    n_test = 3553
    print("Computing artifact counts for all test slices...")
    artifact_counts = [count_artifacts_for_sample(i, ARTIFACT_SEED) for i in range(n_test)]
    dist = {k: artifact_counts.count(k) for k in range(4)}
    print(f"Artifact count distribution: {dist}")

    # ── Load test dataset ─────────────────────────────────────────────────────
    ds = LoDoPaBDataset("test", add_artifacts=True, artifact_seed=ARTIFACT_SEED)

    results = {}

    # ── Model 1: U-Net (large) ─────────────────────────────────────────────────
    print("\nEvaluating U-Net (large)...")
    stages_lg, fbp, cfg_lg = load_model(ROOT / "config/lodopab_ds_baseline_unet_train.json")
    metrics_lg = evaluate_model(stages_lg, fbp, ds, cfg_lg)
    results["unet_large"] = summarise(metrics_lg, artifact_counts)
    print(f"  Overall PSNR: {results['unet_large']['overall']['psnr_mean']:.4f} "
          f"± {results['unet_large']['overall']['psnr_std']:.4f}")

    del stages_lg
    torch.cuda.empty_cache()

    # ── Model 2: Residual dual-filter cascade ─────────────────────────────────
    print("\nEvaluating Residual dual-filter cascade...")
    stages_rd, fbp2, cfg_rd = load_model(ROOT / "config/lodopab_ds_residual_cascade_detach_dual_train.json")
    metrics_rd = evaluate_model(stages_rd, fbp2, ds, cfg_rd)
    results["residual_dual"] = summarise(metrics_rd, artifact_counts)
    print(f"  Overall PSNR: {results['residual_dual']['overall']['psnr_mean']:.4f} "
          f"± {results['residual_dual']['overall']['psnr_std']:.4f}")

    del stages_rd
    torch.cuda.empty_cache()

    # ── Also collect std for ALL models from their existing mean results ───────
    # We only have per-slice arrays for the two above; for the others we use
    # the saved mean values already reported in the thesis.

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = ROOT / "output" / "eval_per_artifact_count.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # ── Print table ─────────────────────────────────────────────────────────
    for model_name, res in results.items():
        print(f"\n=== {model_name} ===")
        ov = res["overall"]
        print(f"Overall (n={ov['n']}): PSNR {ov['psnr_mean']:.4f}±{ov['psnr_std']:.4f}  "
              f"SSIM {ov['ssim_mean']:.4f}±{ov['ssim_std']:.4f}  "
              f"RMSE {ov['rmse_mean']:.5f}±{ov['rmse_std']:.5f}")
        for k in range(4):
            d = res.get(f"artifact_{k}", {})
            if d.get("n", 0) == 0:
                continue
            print(f"  {k} artifacts (n={d['n']:4d}): PSNR {d['psnr_mean']:.4f}±{d['psnr_std']:.4f}  "
                  f"SSIM {d['ssim_mean']:.4f}±{d['ssim_std']:.4f}  "
                  f"RMSE {d['rmse_mean']:.5f}±{d['rmse_std']:.5f}")


if __name__ == "__main__":
    main()
