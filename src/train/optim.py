#optim.py
import torch
import torch.nn as nn
import math
from typing import List, Dict, Optional
try:
    from dion import Muon
except ImportError:
    Muon = None

def get_param_groups_muon(model: nn.Module, muon_lr: float, adamw_lr: float, weight_decay: float) -> List[Dict]:
    """
    Creates parameter groups for Muon optimizer following Microsoft Dion best practices.
    - Hidden 2D weights -> Muon
    - Embeddings, Head, Biases, Norms -> AdamW
    """
    muon_params = []
    adamw_decay_params = []
    adamw_no_decay_params = []
    
    no_muon_ids = set()
    no_decay_ids = set()
    
    # 1. Embeddings -> No Muon, No Decay
    if hasattr(model, 'token_embedding'):
        for p in model.token_embedding.parameters():
            no_muon_ids.add(id(p))
            no_decay_ids.add(id(p))
            
    # 2. Output Head -> No Muon, No Decay
    if hasattr(model, 'output_head'):
        for p in model.output_head.parameters():
            no_muon_ids.add(id(p))
            no_decay_ids.add(id(p))
            
    seen = set()
    norm_types = (getattr(nn, "RMSNorm", tuple()), nn.LayerNorm)
    
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            
            is_no_muon = pid in no_muon_ids
            
            # Check for bias/norm/embedding
            if name.endswith("bias") or isinstance(module, norm_types) or isinstance(module, nn.Embedding):
                is_no_muon = True
                no_decay_ids.add(pid)

            if is_no_muon or param.ndim < 2:
                if pid in no_decay_ids:
                    adamw_no_decay_params.append(param)
                else:
                    adamw_decay_params.append(param)
            else:
                muon_params.append(param)
            
    groups = []
    if muon_params:
        groups.append({
            "params": muon_params,
            "algorithm": "muon",
            "lr": muon_lr,
            "weight_decay": weight_decay
        })
    if adamw_decay_params:
        groups.append({
            "params": adamw_decay_params,
            "algorithm": "adamw", 
            "lr": adamw_lr,
            "weight_decay": weight_decay
        })
    if adamw_no_decay_params:
        groups.append({
            "params": adamw_no_decay_params,
            "algorithm": "adamw",
            "lr": adamw_lr,
            "weight_decay": 0.0
        })
        
    return groups

def build_scheduler(optimizer, num_training_steps: int, warmup_ratio: float = 0.03):
    warmup_steps = max(1, int(num_training_steps * warmup_ratio))
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def build_optimizer_muon(
    model: nn.Module,
    muon_lr: Optional[float] = None,
    adamw_lr: float = 3e-4,
    weight_decay: float = 0.01,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
):
    if Muon is None:
        raise ImportError("dion package is not installed. Please install it with 'pip install git+https://github.com/microsoft/dion.git'")

    if muon_lr is None:
        muon_lr = 0.02 # Default Muon LR is typically higher

    param_groups = get_param_groups_muon(model, muon_lr, adamw_lr, weight_decay)
    
    # Configure optimizer
    # dion.Muon handles mixed algorithms (muon, adamw, lion)
    return Muon(
        param_groups,
        lr=muon_lr, # Global default
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        ns_steps=ns_steps,
    )

def clip_grad_norm(parameters, max_norm: float, norm_type: float = 2.0):
    if isinstance(parameters, nn.Module):
        params = [p for p in parameters.parameters() if p.requires_grad and p.grad is not None]
    else:
        params = [p for p in parameters if p.requires_grad and p.grad is not None]
    if not params:
        return 0.0
    return torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type=norm_type)