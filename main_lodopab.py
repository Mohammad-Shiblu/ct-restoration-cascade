"""
main_lodopab.py — Entry point for LoDoPaB-CT sinogram denoising pipeline.

Run with:
    conda run -n tensor python main_lodopab.py
    conda run -n tensor python main_lodopab.py config/lodopab_train.json

Before first run, download the dataset (~114 GB):
    conda run -n tensor python data_prep/download_lodopab.py
    # or download only the training split first:
    conda run -n tensor python data_prep/download_lodopab.py --parts train
"""

import json
import os
import sys
import torch

from sinogram_trainer.lodopab_trainer import LoDoPaBTrainer
from sinogram_trainer.lodopab_ds_trainer import LoDoPaBDeepSupervisionTrainer

os.environ["TORCH_HOME"] = (
    "/home/shiblu/Project/Masters_thesis/Masters-Thesis/torch_cache"
)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/lodopab_train.json"

    with open(config_path) as f:
        config = json.load(f)

    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    from utils.help import setup_logger
    logger = setup_logger(config)

    logger.info(f"Config : {config_path}")
    logger.info(f"Device : {config['device']}")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")

    
    if "ds" in config_path.lower():
        trainer = LoDoPaBDeepSupervisionTrainer(config=config, logger=logger, test_local=False, max_samples=30000)
    else:
        trainer = LoDoPaBTrainer(config=config, logger=logger, test_local=False)
    trainer.run()


if __name__ == "__main__":
    main()
