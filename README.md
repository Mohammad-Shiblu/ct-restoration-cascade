# Metal Artifact Reduction in CT via Deep Supervision Cascade

Master's Thesis — Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)

This repository contains the full experimental code for the thesis:
**"Deep Learning-Based Metal Artifact Reduction in CT Imaging Using a Deep Supervision Cascade"**

---

## Overview

Metal implants (hip prostheses, dental fillings, surgical clips) cause severe streak and shadow artifacts in X-ray CT through a combination of beam hardening, scatter, and photon starvation. This work frames metal artifact reduction (MAR) as a learned post-processing problem on the sinogram domain and proposes a two-stage deep supervision cascade:

- **Stage 0** — U-Net that denoises the FBP-reconstructed image  
- **Stage 1** — ResUNet that predicts an additive residual correction, supervised directly on the residual error

Physics-informed artifacts (beam hardening, metal streaks, Poisson noise) are injected on-the-fly during training using the real LoDoPaB-CT sinograms. No external MAR dataset is needed.

---

## Results

Full quantitative results are in `notebooks/ct_denoising.ipynb` (Results Summary cell).  
Best model: **Residual Cascade (symmetric, detach)** — PSNR **35.79 dB**, SSIM **0.8724** on 3 553 test slices.

| Method | PSNR (dB) | SSIM |
|---|---|---|
| FBP (noisy input) | ~28.2 | ~0.700 |
| BM3D (σ=0.10) | ~33.9 | ~0.844 |
| RED-CNN | ~33.6 | ~0.840 |
| U-Net (single-stage, large) | ~35.2 | ~0.869 |
| **Residual Cascade (symmetric, detach)** | **35.79** | **0.8724** |

---

## Repository Structure

```
Masters-Thesis/
│
├── main.py                          # Entry point — set CONFIG, run python main.py
│
├── config/                          # One JSON config per experiment
│   ├── lodopab_ds_baseline_unet_small_train.json
│   ├── lodopab_ds_baseline_unet_train.json
│   ├── lodopab_ds_naive_cascade_*.json
│   └── lodopab_ds_residual_cascade_*.json
│
├── models/
│   ├── unet.py                      # U-Net with sigmoid output
│   ├── resunet.py                   # ResUNet with residual blocks
│   └── residual_unet.py             # Alternative ResUNet variant
│
├── trainer/
│   ├── lodopab_ds_trainer.py        # Main trainer (deep supervision cascade)
│   ├── parallel_fbp.py              # Differentiable parallel-beam FBP (torch_radon)
│   └── base.py                      # BaseTrainer: data loaders, save/load, logging
│
├── utils/
│   ├── lodopab_dataset.py           # LoDoPaBDataset — HDF5 loader + on-the-fly artifact injection
│   ├── loss.py                      # Stage0Loss (SSIM+L1), ResidualRefinementLoss
│   ├── metrics.py                   # PSNR, SSIM, RMSE helpers
│   ├── model_inference.py           # load_model, run_inference, plot_results (notebook helpers)
│   └── help.py                      # setup_logger, EarlyStopping
│
├── baselines/
│   ├── bm3d_eval.py                 # BM3D evaluation script
│   └── redcnn/                      # RED-CNN training and evaluation
│       ├── train.py
│       └── test.py
│
├── data_prep/
│   └── download_lodopab.py          # Download LoDoPaB-CT from Zenodo (~55 GB compressed)
│
├── scripts/
│   ├── setup_environment.sh         # One-shot environment setup (conda + torch_radon)
│   ├── build_metal_library.py       # Pre-compute metal-implant sinogram library
│   ├── smoke_check.py               # Quick import / forward-pass sanity check
│   └── test_fbp.py                  # FBP visual sanity check
│
├── notebooks/
│   └── ct_denoising.ipynb           # Inference, visualisation, results summary
│
├── Thesis_report/                   # LaTeX source (compiled on Overleaf)
│
└── requirements.txt
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/Masters-Thesis.git
cd Masters-Thesis
```

### 2. Set up the environment

The script creates a conda environment, installs PyTorch (auto-detects your CUDA version), installs all pip dependencies, and compiles `torch_radon` from source with the required compatibility patches.

```bash
bash scripts/setup_environment.sh            # default env name: ct_denoising
bash scripts/setup_environment.sh myenv      # custom env name
```

**What the script patches in `torch_radon` (matteo-ronchetti/torch-radon, unmaintained):**

| Patch | Reason |
|---|---|
| Restrict GPU archs to sm_80/sm_86 | Older archs removed in CUDA 13 |
| Comment out `CUFFT_INCOMPLETE_PARAMETER_LIST` | Removed from cufft.h in CUDA 13 |
| Replace `torch.rfft`/`irfft` with `torch.fft` API | Removed in PyTorch 2.0 |
| Replace `np.int` with `int` | Removed in NumPy 1.24 |
| Fix Fourier-filter broadcast shape | Complex tensor API change |

**Tested on:** CUDA 12.6, CUDA 13.0 / PyTorch 2.x / Python 3.12 / Linux

> **Windows:** the setup script requires bash (Linux/macOS). On Windows use WSL2.

### 3. Download the dataset

LoDoPaB-CT (Leuschner et al., 2021, [doi:10.1038/s41597-021-00893-z](https://doi.org/10.1038/s41597-021-00893-z)).  
~55 GB compressed, ~114 GB extracted.

```bash
conda activate ct_denoising
python data_prep/download_lodopab.py                           # all splits
python data_prep/download_lodopab.py --parts train             # train only
```

By default the dataset is saved to `medical_image_datasets/lodopab/`.  
Override with the environment variable:

```bash
export LODOPAB_DATA_PATH=/your/path/to/lodopab
```

### 4. Build the metal-implant library

Required for on-the-fly artifact injection during training:

```bash
python scripts/build_metal_library.py --n 200 --device cuda
```

Generates `data/metal/metal_library.npz` (200 metal masks, seeded for reproducibility).

---

## Training

Open `main.py` and set the `CONFIG` variable to the experiment you want to run:

```python
CONFIG = "residual_detach"   # ← change this
```

Available keys:

| Key | Architecture |
|---|---|
| `unet_small` | Single-stage U-Net (small) |
| `unet_large` | Single-stage U-Net (large) |
| `naive_detach` | 2-stage naive cascade, detached |
| `naive_detach_large` | 2-stage naive cascade, detached (large Stage 1) |
| `naive_e2e` | 2-stage naive cascade, end-to-end |
| `residual_detach` | 2-stage residual cascade, detached (symmetric) |
| `residual_detach_dual` | 2-stage residual cascade, detached, dual-domain |
| `residual_detach_asym_ls` | 2-stage residual cascade, detached (large→small) |
| `residual_detach_asym_sl` | 2-stage residual cascade, detached (small→large) |
| `residual_e2e` | 2-stage residual cascade, end-to-end |

Then run:

```bash
conda activate ct_denoising
python main.py
```

For multi-GPU runs (one experiment per GPU):

```bash
CUDA_VISIBLE_DEVICES=0 python main.py > output/gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python main.py > output/gpu1.log 2>&1 &
```

Checkpoints are saved to `checkpoints/<model>/<test_no>/stage_<n>_best.pth`.  
Training logs are saved to `output/<model>/log/`.

---

## Evaluation and Inference

Open `notebooks/ct_denoising.ipynb`. The notebook provides:

- Load a trained model and run inference on test slices
- Visual comparison: FBP | Stage 0 | Stage 1 | Ground Truth
- BM3D and RED-CNN baseline inference
- Full quantitative results table

To load a trained model:

```python
from utils.model_inference import load_model, run_inference, plot_results

stages, fbp, config = load_model("config/lodopab_ds_residual_cascade_detach_train.json")
noisy_t, stage_outputs = run_inference(stages, fbp, sino, config)
fig = plot_results(noisy_t, stage_outputs, gt_t, config, slice_idx=190)
```

---

## Baselines

**BM3D** — tuned on 200 validation samples (σ=0.10):

```bash
python baselines/bm3d_eval.py
```

**RED-CNN** — uses the vendored implementation in `extern/RED-CNN` (not modified):

```bash
python baselines/redcnn/train.py
python baselines/redcnn/test.py
```

---

## Checkpoints

Pre-trained checkpoints are hosted on Hugging Face Hub (link to be added).

---

---

## License

Code: MIT. See `LICENSE`.  
LoDoPaB-CT dataset: CC BY 4.0 ([Zenodo record 3384092](https://zenodo.org/record/3384092)).
