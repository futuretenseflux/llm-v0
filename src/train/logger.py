import time
from typing import Optional, Dict, Any

import wandb


class TrainLogger:
    def __init__(self, project: str, run_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None, mode: str = "online"):
        self.run = wandb.init(project=project, name=run_name, config=config, mode=mode)
        self.start_time = time.time()
        self._last_log_time: float | None = None
        self._last_log_step: int | None = None

    def log_batch(self, batch_idx: int, loss_value: float, step: int):
        wandb.log({"train/loss": loss_value, "train/batch": batch_idx}, step=step)

    def log_train_step(self, *, batch_idx: int, loss_value: float, step: int, total_steps: int = 0):
        now = time.time()

        if self._last_log_time is None:
            self._last_log_time = now
        if self._last_log_step is None:
            self._last_log_step = step

        dt = now - self._last_log_time
        dsteps = step - self._last_log_step
        steps_per_hour = None
        eta_seconds = None
        if dt > 0 and dsteps > 0:
            steps_per_hour = (dsteps / dt) * 3600.0
            if total_steps and total_steps > step:
                eta_seconds = (total_steps - step) / (dsteps / dt)

        if eta_seconds is not None:
            eta_seconds_int = int(max(0.0, eta_seconds))
            hh = eta_seconds_int // 3600
            mm = (eta_seconds_int % 3600) // 60
            ss = eta_seconds_int % 60
            eta_str = f"{hh:02d}:{mm:02d}:{ss:02d}"
        else:
            eta_str = "n/a"

        if steps_per_hour is not None:
            print("Step", step, "loss:", loss_value, "steps/hr:", f"{steps_per_hour:.2f}", "eta:", eta_str)
        else:
            print("Step", step, "loss:", loss_value, "steps/hr:", "n/a", "eta:", eta_str)

        self._last_log_time = now
        self._last_log_step = step
        self.log_batch(batch_idx=batch_idx, loss_value=loss_value, step=step)

    def log_info(self, message: str):
        print(message)

    def finish(self):
        if self.run is not None:
            self.run.finish()