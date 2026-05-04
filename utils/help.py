import os
import logging
from datetime import datetime


def setup_logger(config: dict) -> logging.Logger:
    file_path = os.path.join(config["output_dir"], config["model"], "log")
    os.makedirs(file_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
        file_path,
        f"training_{config['model']}_{config['test_no']}_{timestamp}.log",
    )

    logger = logging.getLogger(config["test_no"])
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in [logging.FileHandler(log_file), logging.StreamHandler()]:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience      = patience
        self.min_delta     = min_delta
        self.counter       = 0
        self.best_loss     = float("inf")
        self.stop_training = False

    def check_early_stop(self, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True
