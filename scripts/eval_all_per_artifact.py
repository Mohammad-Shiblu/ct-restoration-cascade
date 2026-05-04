"""
Compute per-artifact-count metric breakdown for ALL models reported in the thesis.
Outputs JSON to output/eval_all_per_artifact.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import bm3d as _bm3d

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.lodopab_dataset import LoDoPaBDataset, DEFAULT_ARTIFACT_CONFIG
from utils.metrics import compute_psnr, compute_ssim, compute_rmse
from utils.model_inference import load_model, load_redcnn
from trainer.parallel_fbp import DifferentiableParallelFBP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACT_SEED = 0
DR = 1.0


def count_artifacts_for_sample(idx: int, artifact_seed: int = 0) -> int:
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


def summarise(psnr_list, ssim_list, rmse_list, artifact_counts):
    psnr = np.array(psnr_list)
    ssim = np.array(ssim_list)
    rmse = np.array(rmse_list)
    counts = np.array(artifact_counts)
    result = {
        "overall": {
            "n": len(psnr),
            "psnr_mean": float(np.mean(psnr)), "psnr_std": float(np.std(psnr)),
            "ssim_mean": float(np.mean(ssim)), "ssim_std": float(np.std(ssim)),
            "rmse_mean": float(np.mean(rmse)), "rmse_std": float(np.std(rmse)),
        }
    }
    for k in range(4):
        mask = counts == k
        n = int(mask.sum())
        if n == 0:
            result[f"artifact_{k}"] = {"n": 0}
            continue
        result[f"artifact_{k}"] = {
            "n": n,
            "psnr_mean": float(np.mean(psnr[mask])), "psnr_std": float(np.std(psnr[mask])),
            "ssim_mean": float(np.mean(ssim[mask])), "ssim_std": float(np.std(ssim[mask])),
            "rmse_mean": float(np.mean(rmse[mask])), "rmse_std": float(np.std(rmse[mask])),
        }
    return result


def run_unet(stages, fbp, ds, config):
    psnr_list, ssim_list, rmse_list = [], [], []
    num_stages  = config["num_stages"]
    stage1_mode = config.get("stage1_mode", "residual")
    dual_domain = config.get("dual_domain", False)
    with torch.no_grad():
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
            psnr_list.append(compute_psnr(pred, gt_cpu, data_range=DR))
            ssim_list.append(compute_ssim(pred, gt_cpu, data_range=DR))
            r = compute_rmse(pred, gt_cpu)
            rmse_list.append(float(r.item() if hasattr(r, 'item') else r))
            if (idx + 1) % 500 == 0:
                print(f"  {idx+1}/{len(ds)}")
    return psnr_list, ssim_list, rmse_list


def run_redcnn_eval(model, fbp, ds):
    psnr_list, ssim_list, rmse_list = [], [], []
    with torch.no_grad():
        for idx in range(len(ds)):
            sino_t, gt_t = ds[idx]
            sino_t = sino_t.unsqueeze(0).to(DEVICE)
            noisy_t = fbp(sino_t).clamp(0, 1)
            pred = model(noisy_t).clamp(0, 1).cpu()
            gt_cpu = gt_t.unsqueeze(0)
            psnr_list.append(compute_psnr(pred, gt_cpu, data_range=DR))
            ssim_list.append(compute_ssim(pred, gt_cpu, data_range=DR))
            r = compute_rmse(pred, gt_cpu)
            rmse_list.append(float(r.item() if hasattr(r, 'item') else r))
            if (idx + 1) % 500 == 0:
                print(f"  {idx+1}/{len(ds)}")
    return psnr_list, ssim_list, rmse_list


def run_bm3d_eval(fbp, ds, sigma=0.10):
    psnr_list, ssim_list, rmse_list = [], [], []
    with torch.no_grad():
        for idx in range(len(ds)):
            sino_t, gt_t = ds[idx]
            sino_t = sino_t.unsqueeze(0).to(DEVICE)
            noisy_t = fbp(sino_t).clamp(0, 1).cpu()
            noisy_np = noisy_t.squeeze().numpy()
            denoised = _bm3d.bm3d(noisy_np.astype(np.float64), sigma_psd=sigma,
                                   stage_arg=_bm3d.BM3DStages.ALL_STAGES)
            pred = torch.from_numpy(np.clip(denoised, 0, 1).astype(np.float32)).unsqueeze(0).unsqueeze(0)
            gt_cpu = gt_t.unsqueeze(0)
            psnr_list.append(compute_psnr(pred, gt_cpu, data_range=DR))
            ssim_list.append(compute_ssim(pred, gt_cpu, data_range=DR))
            r = compute_rmse(pred, gt_cpu)
            rmse_list.append(float(r.item() if hasattr(r, 'item') else r))
            if (idx + 1) % 500 == 0:
                print(f"  {idx+1}/{len(ds)}")
    return psnr_list, ssim_list, rmse_list


def run_fbp_eval(fbp, ds):
    psnr_list, ssim_list, rmse_list = [], [], []
    with torch.no_grad():
        for idx in range(len(ds)):
            sino_t, gt_t = ds[idx]
            sino_t = sino_t.unsqueeze(0).to(DEVICE)
            noisy_t = fbp(sino_t).clamp(0, 1).cpu()
            gt_cpu = gt_t.unsqueeze(0)
            psnr_list.append(compute_psnr(noisy_t, gt_cpu, data_range=DR))
            ssim_list.append(compute_ssim(noisy_t, gt_cpu, data_range=DR))
            r = compute_rmse(noisy_t, gt_cpu)
            rmse_list.append(float(r.item() if hasattr(r, 'item') else r))
            if (idx + 1) % 500 == 0:
                print(f"  {idx+1}/{len(ds)}")
    return psnr_list, ssim_list, rmse_list


CONFIGS = [
    ("naive_detach",    "config/lodopab_ds_naive_cascade_detach_train.json"),
    ("naive_e2e",       "config/lodopab_ds_naive_cascade_e2e_train.json"),
    ("residual_detach", "config/lodopab_ds_residual_cascade_detach_train.json"),
    ("residual_dual",   "config/lodopab_ds_residual_cascade_detach_dual_train.json"),
    ("unet_small",      "config/lodopab_ds_baseline_unet_small_train.json"),
    ("unet_large",      "config/lodopab_ds_baseline_unet_train.json"),
]


def main():
    n_test = 3553
    print("Computing artifact counts...")
    artifact_counts = [count_artifacts_for_sample(i, ARTIFACT_SEED) for i in range(n_test)]
    dist = {k: artifact_counts.count(k) for k in range(4)}
    print(f"Distribution: {dist}")

    ds = LoDoPaBDataset("test", add_artifacts=True, artifact_seed=ARTIFACT_SEED)

    results = {}

    # FBP
    print("\nEvaluating Corrupted FBP...")
    geom_ds = LoDoPaBDataset("test", None)
    fbp_base = DifferentiableParallelFBP.from_geometry(geom_ds.geometry).to(DEVICE).eval()
    for p in fbp_base.parameters():
        p.requires_grad_(False)
    p, s, r = run_fbp_eval(fbp_base, ds)
    results["fbp"] = summarise(p, s, r, artifact_counts)
    print(f"  FBP PSNR: {results['fbp']['overall']['psnr_mean']:.4f}")

    # BM3D
    print("\nEvaluating BM3D...")
    p, s, r = run_bm3d_eval(fbp_base, ds, sigma=0.10)
    results["bm3d"] = summarise(p, s, r, artifact_counts)
    print(f"  BM3D PSNR: {results['bm3d']['overall']['psnr_mean']:.4f}")

    # RED-CNN
    print("\nEvaluating RED-CNN...")
    redcnn, fbp_rc = load_redcnn()
    p, s, r = run_redcnn_eval(redcnn, fbp_rc, ds)
    results["redcnn"] = summarise(p, s, r, artifact_counts)
    print(f"  RED-CNN PSNR: {results['redcnn']['overall']['psnr_mean']:.4f}")
    del redcnn
    torch.cuda.empty_cache()

    # U-Net and cascade models
    for tag, cfg_path in CONFIGS:
        print(f"\nEvaluating {tag}...")
        stages, fbp, cfg = load_model(ROOT / cfg_path)
        p, s, r = run_unet(stages, fbp, ds, cfg)
        results[tag] = summarise(p, s, r, artifact_counts)
        print(f"  {tag} PSNR: {results[tag]['overall']['psnr_mean']:.4f}")
        del stages
        torch.cuda.empty_cache()

    out_path = ROOT / "output" / "eval_all_per_artifact.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Print summary table
    order = ["fbp", "bm3d", "redcnn", "unet_small", "naive_detach", "naive_e2e",
             "residual_detach", "residual_dual", "unet_large"]
    print(f"\n{'Model':<22} {'n':>5} {'PSNR':>8} {'±':>5} {'SSIM':>7} {'±':>6}  k=0      k=1      k=2      k=3")
    for tag in order:
        if tag not in results:
            continue
        ov = results[tag]["overall"]
        row = f"{tag:<22} {ov['n']:>5} {ov['psnr_mean']:>8.4f} {ov['psnr_std']:>5.2f} {ov['ssim_mean']:>7.4f} {ov['ssim_std']:>6.4f}"
        for k in range(4):
            d = results[tag].get(f"artifact_{k}", {})
            if d.get("n", 0) > 0:
                row += f"  {d['psnr_mean']:>6.2f}"
            else:
                row += "     ---"
        print(row)


if __name__ == "__main__":
    main()
