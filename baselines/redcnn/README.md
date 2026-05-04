# RED-CNN Baseline (Chen et al. 2017)

External literature baseline for low-dose CT denoising. Uses the **published
RED-CNN architecture** (96-filter, 5-conv + 5-deconv residual encoder-decoder,
no padding, MSE loss) — vendored unmodified under `extern/RED-CNN/` from
[SSinyu/RED-CNN](https://github.com/SSinyu/RED-CNN).

## Reference

Chen, H., Zhang, Y., Kalra, M. K., Lin, F., Chen, Y., Liao, P., ... & Wang, G.
*Low-dose CT with a residual encoder-decoder convolutional neural network.*
IEEE Transactions on Medical Imaging, 36(12), 2524–2535, 2017.
<https://doi.org/10.1109/TMI.2017.2715284>

## What is and isn't modified vs upstream

| Component | Source | Notes |
|---|---|---|
| `RED_CNN` model class | `extern/RED-CNN/networks.py` | unchanged |
| Loss (MSE), optimiser (Adam), lr=1e-5, lr×0.5 every 3000 iters | upstream `solver.py` | unchanged |
| 64×64 random patches, `patch_n=10`, `batch_size=16`, 100 epochs | upstream `solver.py` / `loader.py` | unchanged |
| **Data source** | AAPM DICOMs → **LoDoPaB-CT HDF5** | matches every other baseline in the thesis |
| **(input, target) pair** | quarter-dose / full-dose npy → **FBP(artifact-injected sinogram) / clean GT** | upstream's HU normalisation is replaced with LoDoPaB's μ-normalisation ([0, 1], `MU_MAX = 81.35858 m⁻¹`) |
| **FBP** | n/a (upstream loads precomputed npy) | parallel-beam 1000×513 → 362², via `DifferentiableParallelFBP` (torch_radon) on the fly |
| **Test metrics** | upstream HU truncation [-160, 240] | replaced with `data_range = 1.0` in [0, 1] space, matching `baselines/bm3d_eval.py` and the cascade trainer |

The model, loss, optimiser, schedule, patch-extraction strategy, and total
training budget are all unchanged. Only the data path is LoDoPaB-aware.

## Why a separate baseline directory

The earlier in-tree RED-CNN at `models/image_domain/redcnn.py` was an
"inspired-by" rewrite (same-padding, sigmoid/clamp output, custom skip
plumbing). It also failed to learn — the final `clamp(h, 0, 1)` saturated
at init and gradients were exactly zero from epoch 1 onwards. That file has
been deleted; this directory contains the published architecture, run
faithfully, so the citation chain holds.

## End-to-end commands (single GPU)

```bash
# 1) train (~1–2 days on a single RTX 3090; 100 epochs × ~2200 iters/epoch)
conda run -n tensor python baselines/redcnn/train.py

# 2) evaluate on the LoDoPaB artifact-injected test set (TEST_ARTIFACT_SEED=0)
conda run -n tensor python baselines/redcnn/test.py \
    --ckpt checkpoints/baselines/redcnn/REDCNN_220000iter.ckpt
```

Outputs land in `output/baselines/redcnn/`:

```
output/baselines/redcnn/
├── log/                    # training + test logs
├── fig/sample_<idx>.png    # 20 fixed visual comparisons (VIS_SEED=123)
└── metrics.json            # aggregate PSNR / SSIM / RMSE
```

Checkpoints land in `checkpoints/baselines/redcnn/REDCNN_<iter>iter.ckpt`,
mirroring the upstream filename convention.

## Notes / known caveats

- Upstream `extern/RED-CNN/networks.py` was committed with a typo
  (`nn.Conv2D`, capital D — not a real PyTorch symbol; the file would raise
  `AttributeError` on import). Fixed in place by replacing every `nn.Conv2D`
  with `nn.Conv2d`. This is the only change to the vendored file; layer
  shapes, kernel size, padding, and weights are unchanged.
- Upstream uses `padding=0` (no same-padding), so the network output is
  20 px smaller than the input on each side (5×5 conv loses 4 px per layer).
  The 64×64 training patches output as 64×64 because the symmetric
  ConvTranspose2d layers restore the spatial size. At test time, the full
  362×362 input also passes through the network and produces 362×362 output
  for the same reason.
- Each training step samples `patch_n × batch_size = 160` random 64×64
  patches; with 35820 train slices, one epoch ≈ 2200 iterations, total
  ≈ 220 000 iterations — matching the upstream training budget.
- Artifacts are re-randomised every step (`artifact_seed=None` for train),
  matching how every other baseline in this repo is trained.
