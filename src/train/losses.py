#losses.py
import torch
import torch.nn.functional as F

def cross_entropy_shifted(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100):
    # logits: [B, T, V], targets: [B, T]
    # shift: predict token t from inputs up to t-1
    logits = logits[:, :-1, :].contiguous()
    targets = targets[:, 1:].contiguous()
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),ignore_index=ignore_index)
    return loss