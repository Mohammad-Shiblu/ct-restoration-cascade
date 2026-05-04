"""
Fast per-artifact-count breakdown for all GPU models + FBP + RED-CNN.
BM3D is skipped (too slow for per-slice re-evaluation; aggregate already known).
Outputs: output/eval_gpu_models_per_artifact.json
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
from utils.model_inference import load_model, load_redcnn
from trainer.parallel_fbp import DifferentiableParallelFBP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACT_SEED = 0
DR = 1.0


def count_artifacts(idx, artifact_seed=0):
    rng_state = np.random.get_state()
    np.random.seed(artifact_seed + idx)
    cfg = DEFAULT_ARTIFACT_CONFIG
    count = 0
    if cfg.get("motion_blur", False):
        fired = np.random.rand() < cfg.get("motion_blur_prob", 0.5)
        if fired:
            count += 1
            np.random.uniform(*cfg["motion_amplitude"])
            np.random.uniform(*cfg["motion_cycles"])
            np.random.uniform(0, 2 * np.pi)
    if cfg.get("ring", False):
        fired = np.random.rand() < cfg.get("ring_prob", 0.5)
        if fired:
            count += 1
            n_lo, n_hi = cfg["ring_n_cols"]
            n_cols = np.random.randint(n_lo, n_hi + 1)
            for _ in range(n_cols):
                np.random.uniform(*cfg.get("ring_gain", (0.5, 0.95)))
                np.random.rand()
    if cfg.get("metal", False):
        fired = np.random.rand() < cfg.get("metal_prob", 0.3)
        if fired:
            count += 1
    np.random.set_state(rng_state)
    return count


def summarise(psnr_arr, ssim_arr, rmse_arr, counts_arr):
    result = {
        "overall": {
            "n": len(psnr_arr),
            "psnr_mean": float(np.mean(psnr_arr)), "psnr_std": float(np.std(psnr_arr)),
            "ssim_mean": float(np.mean(ssim_arr)), "ssim_std": float(np.std(ssim_arr)),
            "rmse_mean": float(np.mean(rmse_arr)), "rmse_std": float(np.std(rmse_arr)),
        }
    }
    for k in range(4):
        mask = counts_arr == k
        n = int(mask.sum())
        if n == 0:
            result[f"artifact_{k}"] = {"n": 0}
            continue
        result[f"artifact_{k}"] = {
            "n": n,
            "psnr_mean": float(np.mean(psnr_arr[mask])),
            "psnr_std":  float(np.std(psnr_arr[mask])),
            "ssim_mean": float(np.mean(ssim_arr[mask])),
            "ssim_std":  float(np.std(ssim_arr[mask])),
            "rmse_mean": float(np.mean(rmse_arr[mask])),
            "rmse_std":  float(np.std(rmse_arr[mask])),
        }
    return result


@torch.no_grad()
def eval_unet(stages, fbp, ds, config):
    num_stages  = config["num_stages"]
    stage1_mode = config.get("stage1_mode", "residual")
    dual_domain = config.get("dual_domain", False)
    psnr_l, ssim_l, rmse_l = [], [], []
    for idx in range(len(ds)):
        sino_t, gt_t = ds[idx]
        sino_t = sino_t.unsqueeze(0).to(DEVICE)
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
        psnr_l.append(compute_psnr(pred, gt_cpu, DR))
        ssim_l.append(compute_ssim(pred, gt_cpu, DR))
        r = compute_rmse(pred, gt_cpu)
        rmse_l.append(float(r.item() if hasattr(r, 'item') else r))
        if (idx + 1) % 500 == 0:
            print(f"    {idx+1}/3553")
    return np.array(psnr_l), np.array(ssim_l), np.array(rmse_l)


@torch.no_grad()
def eval_fbp(fbp, ds):
    psnr_l, ssim_l, rmse_l = [], [], []
    for idx in range(len(ds)):
        sino_t, gt_t = ds[idx]
        sino_t = sino_t.unsqueeze(0).to(DEVICE)
        noisy_t = fbp(sino_t).clamp(0, 1).cpu()
        gt_cpu = gt_t.unsqueeze(0)
        psnr_l.append(compute_psnr(noisy_t, gt_cpu, DR))
        ssim_l.append(compute_ssim(noisy_t, gt_cpu, DR))
        r = compute_rmse(noisy_t, gt_cpu)
        rmse_l.append(float(r.item() if hasattr(r, 'item') else r))
        if (idx + 1) % 500 == 0:
            print(f"    {idx+1}/3553")
    return np.array(psnr_l), np.array(ssim_l), np.array(rmse_l)


@torch.no_grad()
def eval_redcnn(model, fbp, ds):
    psnr_l, ssim_l, rmse_l = [], [], []
    for idx in range(len(ds)):
        sino_t, gt_t = ds[idx]
        sino_t = sino_t.unsqueeze(0).to(DEVICE)
        noisy_t = fbp(sino_t).clamp(0, 1)
        pred = model(noisy_t).clamp(0, 1).cpu()
        gt_cpu = gt_t.unsqueeze(0)
        psnr_l.append(compute_psnr(pred, gt_cpu, DR))
        ssim_l.append(compute_ssim(pred, gt_cpu, DR))
        r = compute_rmse(pred, gt_cpu)
        rmse_l.append(float(r.item() if hasattr(r, 'item') else r))
        if (idx + 1) % 500 == 0:
            print(f"    {idx+1}/3553")
    return np.array(psnr_l), np.array(ssim_l), np.array(rmse_l)


CONFIGS = [
    ("unet_small",      "config/lodopab_ds_baseline_unet_small_train.json"),
    ("unet_large",      "config/lodopab_ds_baseline_unet_train.json"),
    ("naive_detach",    "config/lodopab_ds_naive_cascade_detach_train.json"),
    ("naive_e2e",       "config/lodopab_ds_naive_cascade_e2e_train.json"),
    ("residual_detach", "config/lodopab_ds_residual_cascade_detach_train.json"),
    ("residual_dual",   "config/lodopab_ds_residual_cascade_detach_dual_train.json"),
]


def main():
    print("Computing artifact counts for 3553 test slices...")
    counts = np.array([count_artifacts(i, ARTIFACT_SEED) for i in range(3553)])
    print(f"  k=0:{(counts==0).sum()}  k=1:{(counts==1).sum()}  k=2:{(counts==2).sum()}  k=3:{(counts==3).sum()}")

    ds  = LoDoPaBDataset("test", add_artifacts=True, artifact_seed=ARTIFACT_SEED)
    results = {}

    # Shared FBP
    geom_ds = LoDoPaBDataset("test", None)
    fbp = DifferentiableParallelFBP.from_geometry(geom_ds.geometry).to(DEVICE).eval()
    for p in fbp.parameters():
        p.requires_grad_(False)

    print("\n[1/9] Corrupted FBP...")
    p, s, r = eval_fbp(fbp, ds)
    results["fbp"] = summarise(p, s, r, counts)
    print(f"  Overall PSNR {results['fbp']['overall']['psnr_mean']:.4f}")

    print("\n[2/9] RED-CNN...")
    redcnn, fbp_rc = load_redcnn()
    p, s, r = eval_redcnn(redcnn, fbp_rc, ds)
    results["redcnn"] = summarise(p, s, r, counts)
    print(f"  Overall PSNR {results['redcnn']['overall']['psnr_mean']:.4f}")
    del redcnn; torch.cuda.empty_cache()

    for i, (tag, cfg_path) in enumerate(CONFIGS, start=3):
        print(f"\n[{i}/9] {tag}...")
        stages, fbp_m, cfg = load_model(ROOT / cfg_path)
        p, s, r = eval_unet(stages, fbp_m, ds, cfg)
        results[tag] = summarise(p, s, r, counts)
        print(f"  Overall PSNR {results[tag]['overall']['psnr_mean']:.4f}")
        del stages; torch.cuda.empty_cache()

    out = ROOT / "output" / "eval_gpu_models_per_artifact.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")

    # ── Pretty print ──────────────────────────────────────────────────────────
    labels = {
        "fbp":             "Corrupted FBP",
        "redcnn":          "RED-CNN",
        "unet_small":      "U-Net (small)",
        "unet_large":      "U-Net (large)",
        "naive_detach":    "Naive indep.",
        "naive_e2e":       "Naive e2e",
        "residual_detach": "Residual",
        "residual_dual":   "Residual + dual",
    }
    print(f"\n{'Method':<22} {'Overall':>8}  {'k=0':>7}  {'k=1':>7}  {'k=2':>7}  {'k=3':>7}")
    print("-" * 72)
    for tag, lbl in labels.items():
        if tag not in results:
            continue
        ov = results[tag]["overall"]
        row = f"{lbl:<22} {ov['psnr_mean']:>8.2f}"
        for k in range(4):
            d = results[tag].get(f"artifact_{k}", {})
            row += f"  {d['psnr_mean']:>7.2f}" if d.get("n", 0) > 0 else "      ---"
        print(row)


if __name__ == "__main__":
    main()
