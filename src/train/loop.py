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
    use_amp: bool = True, tokens_elapsed: int = 0, total_steps: int = 0, checkpoint_dir: str = "checkpoints"
):
    model.train()
    global_step = 0
    total_loss = 0.0
    epoch = 0
    scaler = None
    
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if use_amp and "cuda" in str(device).lower():
        scaler = amp.GradScaler(enabled=True)

    for batch_idx, (input_ids, targets) in enumerate(train_loader):
        input_ids, targets = input_ids.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if "cuda" in str(device).lower() else "cpu", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(input_ids)
            loss = cross_entropy_shifted(logits=logits, targets=targets)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if max_grad_norm is not None:
            clip_grad_norm(model, max_grad_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if sampler is not None:
            new_probs = get_sampling_ratios(tokens_elapsed)
            sampler.set_probs(new_probs)
        # Calculate tokens processed in this batch
        batch_tokens = input_ids.numel()  # total tokens in batch
        tokens_elapsed += batch_tokens
        
        total_loss += loss.item()
        if logger is not None and batch_idx % log_every == 0:
            logger.log_batch(epoch=epoch + 1, batch_idx=batch_idx, loss_value=loss.item(), step=global_step)
        
        # Checkpoint saving every 20k steps
        checkpoint_interval_steps = 20000  # 20 thousand steps
        if global_step > 0 and global_step % checkpoint_interval_steps == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{tokens_elapsed//1_000_000_000}B_{tokens_elapsed%1_000_000_000}M.pt")
            save_checkpoint(model, optimizer, scheduler, tokens_elapsed, global_step, checkpoint_path)
            if logger is not None:
                logger.log_info(f"Checkpoint saved at step {global_step} with {tokens_elapsed//1_000_000_000}B tokens: {checkpoint_path}")
        
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
