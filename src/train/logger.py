import time
from typing import Optional, Dict, Any

import wandb


class TrainLogger:
    def __init__(self, project: str, run_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None, mode: str = "online"):
        self.run = wandb.init(project=project, name=run_name, config=config, mode=mode)
        self.start_time = time.time()

    def log_batch(self, epoch: int, batch_idx: int, loss_value: float, step: int):
        wandb.log({"train/loss": loss_value, "train/epoch": epoch, "train/batch": batch_idx}, step=step)

    def log_epoch(self, epoch: int, avg_loss: float):
        elapsed = time.time() - self.start_time
        wandb.log({"train/epoch_avg_loss": avg_loss, "train/epoch": epoch, "time/elapsed_sec": elapsed})

    def finish(self):
        if self.run is not None:
            self.run.finish()