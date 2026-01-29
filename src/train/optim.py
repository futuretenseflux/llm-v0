#optim.py
import torch
import torch.nn as nn
import math
from typing import List, Dict

def get_param_groups(model: nn.Module, weight_decay: float) -> List[Dict]:
    decay_params = []
    no_decay_params = []
    seen = set()
    norm_types = (getattr(nn, "RMSNorm", tuple()),)
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            if name.endswith("bias"):
                no_decay_params.append(param)
            elif isinstance(module, norm_types):
                no_decay_params.append(param)
            elif isinstance(module, nn.Linear):
                decay_params.append(param)
            elif isinstance(module, nn.Embedding):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

def build_scheduler(optimizer, num_training_steps: int, warmup_ratio: float = 0.03):
    warmup_steps = max(1, int(num_training_steps * warmup_ratio))
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def build_optimizer(model: nn.Module, lr: float, weight_decay: float, betas=(0.9, 0.95), eps: float = 1e-8):
    param_groups = get_param_groups(model, weight_decay)
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)

def clip_grad_norm(parameters, max_norm: float, norm_type: float = 2.0):
    if isinstance(parameters, nn.Module):
        params = [p for p in parameters.parameters() if p.requires_grad and p.grad is not None]
    else:
        params = [p for p in parameters if p.requires_grad and p.grad is not None]
    if not params:
        return 0.0
    return torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type=norm_type)