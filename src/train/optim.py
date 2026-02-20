#optim.py
import torch
import torch.nn as nn
import math
from typing import List, Dict, Optional

def get_param_groups_muon(model: nn.Module, muon_lr: float, adamw_lr: float, weight_decay: float) -> List[Dict]:
    muon_params = []
    adamw_decay_params = []
    adamw_no_decay_params = []
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
            
            if param.ndim >= 2:
                muon_params.append(param)
            elif name.endswith("bias") or isinstance(module, norm_types):
                adamw_no_decay_params.append(param)
            elif isinstance(module, nn.Embedding):
                adamw_no_decay_params.append(param)
            else:
                adamw_decay_params.append(param)
    
    param_groups = []
    if muon_params:
        param_groups.append({
            "params": muon_params,
            "lr": muon_lr,
            "weight_decay": weight_decay,
            "optimizer_class": "muon"
        })
    if adamw_decay_params:
        param_groups.append({
            "params": adamw_decay_params,
            "lr": adamw_lr,
            "weight_decay": weight_decay,
            "optimizer_class": "adamw"
        })
    if adamw_no_decay_params:
        param_groups.append({
            "params": adamw_no_decay_params,
            "lr": adamw_lr,
            "weight_decay": 0.0,
            "optimizer_class": "adamw"
        })
    
    return param_groups

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

def build_optimizer_muon(
    model: nn.Module,
    muon_lr: Optional[float] = None,
    adamw_lr: float = 3e-4,
    weight_decay: float = 0.01,
    betas=(0.9, 0.95),
    eps: float = 1e-8,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    adjust_lr_fn: str = "match_rms_adamw",
):
    if muon_lr is None:
        muon_lr = adamw_lr
    param_groups = get_param_groups_muon(model, muon_lr, adamw_lr, weight_decay)
    
    muon_groups = [g for g in param_groups if g.get("optimizer_class") == "muon"]
    adamw_groups = [g for g in param_groups if g.get("optimizer_class") == "adamw"]
    
    for g in muon_groups:
        g.pop("optimizer_class", None)
    for g in adamw_groups:
        g.pop("optimizer_class", None)
    
    if muon_groups and adamw_groups:
        muon_opt = torch.optim.Muon(muon_groups, lr=muon_lr, weight_decay=weight_decay, 
                                    momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, eps=eps, adjust_lr_fn=adjust_lr_fn)
        adamw_opt = torch.optim.AdamW(adamw_groups, lr=adamw_lr, betas=betas, eps=eps)
        return CombinedOptimizer([muon_opt, adamw_opt])
    elif muon_groups:
        return torch.optim.Muon(muon_groups, lr=muon_lr, weight_decay=weight_decay,
                               momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, eps=eps, adjust_lr_fn=adjust_lr_fn)
    else:
        return torch.optim.AdamW(adamw_groups, lr=adamw_lr, betas=betas, eps=eps)

class CombinedOptimizer:
    def __init__(self, optimizers):
        self.optimizers = optimizers
        self._param_groups = []
        for opt in optimizers:
            self._param_groups.extend(opt.param_groups)
    
    @property
    def param_groups(self):
        combined = []
        for opt in self.optimizers:
            combined.extend(opt.param_groups)
        return combined
    
    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)
    
    def step(self, closure=None):
        loss = None
        for opt in self.optimizers:
            if closure is not None:
                loss = opt.step(closure)
            else:
                opt.step()
        return loss
    
    def state_dict(self):
        return {f"optimizer_{i}": opt.state_dict() for i, opt in enumerate(self.optimizers)}
    
    def load_state_dict(self, state_dict):
        for i, opt in enumerate(self.optimizers):
            opt.load_state_dict(state_dict[f"optimizer_{i}"])

def clip_grad_norm(parameters, max_norm: float, norm_type: float = 2.0):
    if isinstance(parameters, nn.Module):
        params = [p for p in parameters.parameters() if p.requires_grad and p.grad is not None]
    else:
        params = [p for p in parameters if p.requires_grad and p.grad is not None]
    if not params:
        return 0.0
    return torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type=norm_type)