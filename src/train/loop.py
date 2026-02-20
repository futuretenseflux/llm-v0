#loop.py
from .losses import cross_entropy_shifted
from .optim import clip_grad_norm
from .logger import TrainLogger
from .checkpointer import maybe_save_checkpoint
import torch
from torch import amp
from ..data.pretraining.training.sampling_ratio_generator import get_sampling_ratios
from typing import Optional, Set

try:
    from torch.backends.cuda import sdp_kernel
except Exception:  # pragma: no cover
    sdp_kernel = None

def train_loop(
    model, train_loader, optimizer, device, scheduler=None, sampler=None, max_grad_norm=None, log_every=10, logger: TrainLogger | None = None,
    use_amp: bool = True, tokens_elapsed: int = 0, total_steps: int = 0, checkpoint_dir: str = "checkpoints", grad_accum_steps: int = 1,
    micro_batch_size: int | None = None,
    checkpoint_steps: Optional[Set[int]] = None,
):
    model.train()
    if checkpoint_steps is None:
        raise ValueError("checkpoint_steps must be provided")
    global_step = 0
    total_loss = 0.0
    scaler = None
    sdpa_flash_probe_done = False

    if use_amp and "cuda" in str(device).lower():
        # scaler = amp.GradScaler(enabled=True)
        pass

    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")

    optimizer.zero_grad(set_to_none=True)
    mark_step_begin = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)

    for batch_idx, (input_ids, targets) in enumerate(train_loader):
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

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
                if (not sdpa_flash_probe_done) and (sdp_kernel is not None) and ("cuda" in str(device).lower()):
                    try:
                        with sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False):
                            if mark_step_begin is not None:
                                mark_step_begin()
                            logits = model(micro_input)
                        print("SDPA FlashAttention probe: usable (Flash forced for 1 forward pass)")
                    except Exception as e:
                        print(f"SDPA FlashAttention probe: NOT usable (Flash forced) | {type(e).__name__}: {e}")
                        if mark_step_begin is not None:
                            mark_step_begin()
                        logits = model(micro_input)
                    sdpa_flash_probe_done = True
                else:
                    if mark_step_begin is not None:
                        mark_step_begin()
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
            logger.log_train_step(batch_idx=batch_idx, loss_value=loss.item(), step=global_step, total_steps=total_steps)
        
        if is_accum_boundary:
            checkpoint_path = maybe_save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                tokens_elapsed=tokens_elapsed,
                global_step=global_step,
                checkpoint_dir=checkpoint_dir,
                checkpoint_steps=checkpoint_steps,
            )
            if checkpoint_path is not None and logger is not None:
                logger.log_info(f"Checkpoint saved at step {global_step} with {tokens_elapsed//1_000_000_000}B tokens: {checkpoint_path}")

        if is_accum_boundary:
            global_step += 1
