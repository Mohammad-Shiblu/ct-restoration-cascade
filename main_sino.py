"""
main_sino.py — Entry point for the sinogram-domain denoising pipeline.

Run with:
    conda run -n tensor python main_sino.py
    conda run -n tensor python main_sino.py config/sinogram_train.json

Keeps the sinogram pipeline entirely separate from the image-domain
pipeline (main.py / train/ / utils/dataset.py).
"""

import json
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault(
    "TORCH_HOME",
    str(Path(__file__).parent / "torch_cache"),
)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/sinogram_train.json"

    with open(config_path) as f:
        config = json.load(f)

    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Logger (reuse the same helper as the image pipeline)
    from utils.help import setup_logger
    logger = setup_logger(config)

    logger.info(f"Config: {config_path}")
    logger.info(f"Device: {config['device']}")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")

    from sinogram_trainer.trainer import SinogramTrainer
    trainer = SinogramTrainer(config=config, logger=logger, test_local=False)
    trainer.run()


if __name__ == "__main__":
    main()
