import os
from pathlib import Path
from typing import Optional

import torch


def format_checkpoint_filename(tokens_elapsed: int) -> str:
    return f"checkpoint_{tokens_elapsed//1_000_000_000}B_{tokens_elapsed%1_000_000_000}M.pt"


def save_checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    tokens_elapsed: int,
    global_step: int,
    checkpoint_path: str,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "tokens_elapsed": tokens_elapsed,
        "global_step": global_step,
    }
    torch.save(checkpoint, checkpoint_path)


def maybe_save_checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    tokens_elapsed: int,
    global_step: int,
    checkpoint_dir: str,
    interval_steps: int,
) -> Optional[str]:
    if interval_steps <= 0:
        return None

    if global_step <= 0 or (global_step % interval_steps) != 0:
        return None

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, format_checkpoint_filename(tokens_elapsed))
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        tokens_elapsed=tokens_elapsed,
        global_step=global_step,
        checkpoint_path=checkpoint_path,
    )
    return checkpoint_path
