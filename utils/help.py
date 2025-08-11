import torch.nn as nn
import os
import torchvision.models as models
from datetime import datetime
import logging

# Set up logger function
def setup_logger(config):
    file_path = os.path.join(config["output_dir"], config['model'], "log")

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(file_path, f"training_{config['model']}_{config['test_no']}_{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level= logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            # logging.streamHandler()
        ]
    )
    return logging.getLogger(__name__)





def generate_resutls():
    # this will generate a curve that will combine the result of different pdf and generate a comparison curve based on that.
    pass


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.stop_training = False

    def check_early_stop(self, val_loss):
        if val_loss < self.best_loss -self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True
    
