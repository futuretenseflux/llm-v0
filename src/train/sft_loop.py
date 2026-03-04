import time
from typing import Optional

import torch

from .losses import cross_entropy_shifted
from .optim import clip_grad_norm


def sft_train_loop(
    *,
    model,
    train_loader,
    optimizer,
    device,
    scheduler=None,
    micro_batch_size: Optional[int] = None,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
    log_every: int = 10,
    pad_token_id: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_interval_tokens: int = 50_000_000,
    checkpoint_prefix: str = "sft",
    wandb_run=None,
) -> None:
    model.train()

    global_step = 0
    start_time = time.time()

    tokens_seen = 0
    next_checkpoint_tokens = int(checkpoint_interval_tokens) if checkpoint_dir is not None else None

    optimizer.zero_grad(set_to_none=True)
    mark_step_begin = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)

    print("Epoch 1/1")

    for batch_idx, (input_ids, targets) in enumerate(train_loader):
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if pad_token_id is None:
            tokens_seen += int(input_ids.numel())
        else:
            tokens_seen += int((input_ids != int(pad_token_id)).sum().item())

        batch_size = int(input_ids.size(0))
        mb = int(micro_batch_size) if micro_batch_size is not None else batch_size
        if mb < 1:
            raise ValueError("micro_batch_size must be >= 1")

        num_micro = (batch_size + mb - 1) // mb
        batch_loss = 0.0

        for micro_start in range(0, batch_size, mb):
            micro_end = min(batch_size, micro_start + mb)
            micro_input = input_ids[micro_start:micro_end]
            micro_targets = targets[micro_start:micro_end]

            with torch.autocast(
                device_type="cuda" if "cuda" in str(device).lower() else "cpu",
                dtype=torch.bfloat16,
                enabled=bool(use_amp),
            ):
                if mark_step_begin is not None:
                    mark_step_begin()
                logits = model(micro_input)
                loss_micro = cross_entropy_shifted(logits=logits, targets=micro_targets)

            batch_loss += float(loss_micro.item())
            loss = loss_micro * (micro_input.size(0) / batch_size)
            loss.backward()

        avg_batch_loss = batch_loss / max(1, num_micro)

        if max_grad_norm is not None:
            clip_grad_norm(model, float(max_grad_norm))

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if scheduler is not None:
            scheduler.step()

        global_step += 1

        if checkpoint_dir is not None and next_checkpoint_tokens is not None and tokens_seen >= next_checkpoint_tokens:
            ckpt_path = f"{checkpoint_dir}/{checkpoint_prefix}_tok{next_checkpoint_tokens}.pt"
            torch.save(model.state_dict(), ckpt_path)
            if wandb_run is not None:
                try:
                    wandb_run.log(
                        {
                            "checkpoint/tokens": int(next_checkpoint_tokens),
                        },
                        step=int(global_step),
                    )
                except Exception:
                    pass
            while tokens_seen >= next_checkpoint_tokens:
                next_checkpoint_tokens += int(checkpoint_interval_tokens)

        if log_every and (global_step % int(log_every) == 0):
            elapsed = time.time() - start_time
            steps_per_sec = global_step / max(elapsed, 1e-9)
            lr = None
            if scheduler is not None:
                try:
                    lr = float(scheduler.get_last_lr()[0])
                except Exception:
                    lr = None

            msg = f"Step {global_step} | loss {avg_batch_loss:.4f}"
            if lr is not None:
                msg += f" | lr {lr:.3e}"
            msg += f" | {steps_per_sec:.2f} steps/s"
            print(msg)

            if wandb_run is not None:
                try:
                    metrics = {
                        "train/loss": float(avg_batch_loss),
                        "train/steps_per_sec": float(steps_per_sec),
                        "train/tokens_seen": int(tokens_seen),
                    }
                    if lr is not None:
                        metrics["train/lr"] = float(lr)
                    wandb_run.log(metrics, step=int(global_step))
                except Exception:
                    pass
