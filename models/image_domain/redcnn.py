"""
RED-CNN
=======
Residual Encoder-Decoder CNN for low-dose CT denoising.

Reference
---------
    Chen, H., Zhang, Y., Kalra, M. K., Lin, F., Chen, Y., Liao, P., ... & Wang, G.
    "Low-dose CT with a residual encoder-decoder convolutional neural network."
    IEEE Transactions on Medical Imaging, 36(12), 2524–2535, 2017.
    https://doi.org/10.1109/TMI.2017.2715284

Architecture
------------
    5 convolutional layers (encoder)  → 5 deconvolutional layers (decoder)
    Symmetric shortcut connections:
        enc_1 output → added before dec_5 activation
        enc_2 output → added before dec_4 activation
        enc_3 output → added before dec_3 activation
    All layers: 96 filters, 5×5 kernels, same-padding (spatial size preserved).
    Output: clamped to [0, 1].

Adaptation from the original paper
------------------------------------
    The paper uses 55×55 input patches without padding (size shrinks through
    the network).  Here we use same-padding throughout so the full 362×362
    LoDoPaB image is processed in a single forward pass — no patch extraction
    or stitching needed.
"""

import torch
import torch.nn as nn


class REDCNN(nn.Module):
    """Residual Encoder-Decoder CNN (Chen et al. 2017).

    Parameters
    ----------
    num_filters : int
        Number of feature maps in every layer (default: 96, as in the paper).
    kernel_size : int
        Convolutional kernel size (default: 5, as in the paper).
    num_layers : int
        Number of encoder layers (= number of decoder layers).  Default: 5.
    in_channels : int
        Input channels (1 for single-channel CT images).
    """

    def __init__(
        self,
        num_filters: int = 96,
        kernel_size: int = 5,
        num_layers:  int = 5,
        in_channels: int = 1,
    ):
        super().__init__()
        assert num_layers >= 3, "Need at least 3 layers for the three skip connections."

        self.num_layers = num_layers
        pad = kernel_size // 2   # same-padding for odd kernel sizes

        # ── Encoder ──────────────────────────────────────────────────────────
        enc = []
        enc.append(nn.Conv2d(in_channels, num_filters, kernel_size, padding=pad))
        for _ in range(num_layers - 1):
            enc.append(nn.Conv2d(num_filters, num_filters, kernel_size, padding=pad))
        self.encoder = nn.ModuleList(enc)

        # ── Decoder ──────────────────────────────────────────────────────────
        dec = []
        for _ in range(num_layers - 1):
            dec.append(nn.ConvTranspose2d(num_filters, num_filters, kernel_size, padding=pad))
        # Last deconv maps back to 1 channel (CT image)
        dec.append(nn.ConvTranspose2d(num_filters, in_channels, kernel_size, padding=pad))
        self.decoder = nn.ModuleList(dec)

        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  (B, 1, H, W) — noisy FBP reconstruction in [0, 1].

        Returns
        -------
        Tensor  (B, 1, H, W) — denoised image clamped to [0, 1].
        """
        n = self.num_layers

        # ── Encoder pass — save activations for skip connections ─────────────
        enc_out = []
        h = x
        for conv in self.encoder:
            h = self.relu(conv(h))
            enc_out.append(h)

        # ── Decoder pass ─────────────────────────────────────────────────────
        # Shortcuts (paper Fig. 1):
        #   enc_1 → before last  decoder layer    (index n-1)
        #   enc_2 → before (n-2)-th decoder layer
        #   enc_3 → before (n-3)-th decoder layer
        h = enc_out[-1]   # bottleneck feature (output of last encoder layer)
        for i, deconv in enumerate(self.decoder):
            dec_layer_idx = i          # 0 … n-1
            # Determine which encoder output (if any) to add as skip
            # enc_k connects to dec_(n-k-1):
            #   k=0 (enc_1) → dec_(n-1)  (last decoder)
            #   k=1 (enc_2) → dec_(n-2)
            #   k=2 (enc_3) → dec_(n-3)
            for k in range(3):
                if dec_layer_idx == n - 1 - k:
                    h = h + enc_out[k]
                    break

            if i < len(self.decoder) - 1:
                h = self.relu(deconv(h))
            else:
                h = deconv(h)          # no ReLU before final output

        return torch.clamp(h, 0.0, 1.0)
