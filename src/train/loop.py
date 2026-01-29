#loop.py
from .losses import cross_entropy_shifted
from .optim import clip_grad_norm
from .logger import TrainLogger
import torch
from torch import amp

def train_loop(
    model, train_loader, optimizer, device, num_epochs, scheduler=None, max_grad_norm=None, log_every=100, logger: TrainLogger | None = None,
    use_amp: bool = True
):
    model.train()
    global_step = 0
    scaler = None
    if use_amp and "cuda" in str(device).lower():
        scaler = amp.GradScaler(enabled=True)
    for epoch in range(num_epochs):
        total_loss = 0
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
            total_loss += loss.item()
            if logger is not None and batch_idx % log_every == 0:
                logger.log_batch(epoch=epoch + 1, batch_idx=batch_idx, loss_value=loss.item(), step=global_step)
            global_step += 1
        avg_loss = total_loss / len(train_loader)
        if logger is not None:
            logger.log_epoch(epoch=epoch + 1, avg_loss=avg_loss)