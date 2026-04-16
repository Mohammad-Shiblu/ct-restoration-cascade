# Masters-Thesis

## Project Structure

```
Masters-Thesis/
│
├── base.py                          # Shared base class: get_loader, save/load_model,
│                                    #   set_train_epoch, plot_loss_curves, abstract stubs
│
├── models/
│   ├── image_domain/
│   │   ├── unet.py                  # Standard U-Net
│   │   ├── resunet.py               # U-Net with residual blocks (ResUNet)
│   │   ├── residual_unet.py         # Alternative residual U-Net (ImprovedUNet)
│   │   └── munet.py                 # Cascaded U-Net variants
│   └── sinogram_domain/
│       └── dncnn.py                 # Lightweight DnCNN-style residual corrector
│
├── image_trainer/
│   ├── generic_trainer.py           # UNetTrainer — single U-Net baseline
│   ├── munet_trainer.py             # CascadedUnetTrainer — joint multi-stage training
│   └── separate_trainer.py         # ProgressiveCascadedTrainer — primary trainer
│
├── sinogram_trainer/
│   ├── trainer.py                   # SinogramTrainer — end-to-end sinogram denoising
│   └── fbp.py                       # DifferentiableFBP + radon_fbp (torch_radon)
│
├── utils/
│   ├── ct_image_dataset.py          # CtDataset — paired CT image loader with patch support
│   ├── sinogram_dataset.py          # SinogramDataset — paired fan-beam sinogram loader
│   ├── loss.py                      # Loss functions (SSIMLoss, LPIPS, Stage0Loss, etc.)
│   ├── metrics.py                   # PSNR, SSIM, RMSE with optional body mask
│   └── help.py                      # setup_logger, EarlyStopping
│
├── data_prep/                       # One-time data preparation scripts (not part of pipeline)
│   ├── image_data_prep.py           # CT image pair generation from DICOM
│   └── sinogram_data_prep.py        # Physics-informed sinogram pair generation
│
├── config/
│   ├── train.json                   # Image-domain training config
│   ├── sinogram_train.json          # Sinogram-domain training config
│   ├── data_prep.json               # Image data prep config
│   └── projection_prep.json         # Sinogram data prep config
│
├── extern/
│   └── helix2fan/                   # Vendored helical-to-fan-beam rebinning (Apache 2.0)
│
├── main.py                          # Entry point — image-domain training
└── main_sino.py                     # Entry point — sinogram-domain training
```
