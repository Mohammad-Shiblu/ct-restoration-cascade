import json
import torch
from utils.help import setup_logger
from image_trainer.generic_trainer import UNetTrainer
from image_trainer.munet_trainer import CascadedUnetTrainer
from image_trainer.separate_trainer import ProgressiveCascadedTrainer
import os
os.environ['TORCH_HOME'] = '/home/shiblu/Project/Masters_thesis/Masters-Thesis/torch_cache'


def main():
    with open("config/train.json") as f:
        config = json.load(f)
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger = setup_logger(config)
    logger.info("Starting Training  with config:")
    for key, value in list(config.items())[1:]:
        logger.info(f"{key}: {value}")


   
    # system = CascadedUnetTrainer(config=config, logger=logger, test_local=False)
    # system = UNetTrainer(config=config, logger=logger, test_local=False)
    system = ProgressiveCascadedTrainer(config=config, logger=logger, test_local=False)

    system.run()


if __name__ == "__main__":
    main()



