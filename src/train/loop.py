#loop.py
from .losses import cross_entropy_shifted
from .optim import clip_grad_norm
from .logger import TrainLogger
import torch
from torch import amp
import os
from pathlib import Path
from ..data.pretraining.training.sampling_ratio_generator import get_sampling_ratios

def train_loop(
    model, train_loader, optimizer, device, scheduler=None, sampler=None, max_grad_norm=None, log_every=100, logger: TrainLogger | None = None,
    use_amp: bool = True, tokens_elapsed: int = 0, total_steps: int = 0, checkpoint_dir: str = "checkpoints", grad_accum_steps: int = 1,
    micro_batch_size: int | None = None
):
    model.train()
    global_step = 0
    total_loss = 0.0
    scaler = None
    
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if use_amp and "cuda" in str(device).lower():
        # scaler = amp.GradScaler(enabled=True)
        pass

    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (input_ids, targets) in enumerate(train_loader):
        input_ids, targets = input_ids.to(device), targets.to(device)

        batch_size = int(input_ids.size(0))
        mb = int(micro_batch_size) if micro_batch_size is not None else batch_size
        if mb < 1:
            raise ValueError("micro_batch_size must be >= 1")

        num_micro = (batch_size + mb - 1) // mb
        for micro_start in range(0, batch_size, mb):
            micro_end = min(batch_size, micro_start + mb)
            micro_input = input_ids[micro_start:micro_end]
            micro_targets = targets[micro_start:micro_end]

            with torch.autocast(device_type="cuda" if "cuda" in str(device).lower() else "cpu", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(micro_input)
                loss = cross_entropy_shifted(logits=logits, targets=micro_targets)
                loss = loss / (grad_accum_steps * num_micro)

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        is_accum_boundary = ((batch_idx + 1) % grad_accum_steps) == 0
        if is_accum_boundary:
            if max_grad_norm is not None:
                clip_grad_norm(model, max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        if sampler is not None:
            new_probs = get_sampling_ratios(tokens_elapsed)
            sampler.set_probs(new_probs)
        # Calculate tokens processed in this batch
        batch_tokens = input_ids.numel()  # total tokens in batch
        tokens_elapsed += batch_tokens
        
        total_loss += loss.item()
        if logger is not None and is_accum_boundary and global_step % log_every == 0:
            print("Step", global_step, "loss:", loss.item())
            logger.log_batch(batch_idx=batch_idx, loss_value=loss.item(), step=global_step)
        
        # Checkpoint saving every 20k steps
        checkpoint_interval_steps = 20000  # 20 thousand steps
        if is_accum_boundary and global_step > 0 and global_step % checkpoint_interval_steps == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{tokens_elapsed//1_000_000_000}B_{tokens_elapsed%1_000_000_000}M.pt")
            save_checkpoint(model, optimizer, scheduler, tokens_elapsed, global_step, checkpoint_path)
            if logger is not None:
                logger.log_info(f"Checkpoint saved at step {global_step} with {tokens_elapsed//1_000_000_000}B tokens: {checkpoint_path}")

        if is_accum_boundary:
            global_step += 1

def save_checkpoint(model, optimizer, scheduler, tokens_elapsed, global_step, checkpoint_path):
    """Save model checkpoint with all necessary state."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'tokens_elapsed': tokens_elapsed,
        'global_step': global_step,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
