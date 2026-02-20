#losses.py
import torch
import torch.nn.functional as F

def cross_entropy_shifted(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100):
    # logits: [B, T, V], targets: [B, T]
    # targets are already shifted by the dataset, so logits[t] predicts targets[t]
    if logits.ndim != 3:
        raise ValueError(f"Expected logits with shape [B, T, V], got {tuple(logits.shape)}")
    if targets.ndim != 2:
        raise ValueError(f"Expected targets with shape [B, T], got {tuple(targets.shape)}")
    if logits.size(0) != targets.size(0):
        raise ValueError(f"Batch size mismatch: logits B={int(logits.size(0))} targets B={int(targets.size(0))}")
    if logits.size(1) != targets.size(1):
        raise ValueError(f"Sequence length mismatch: logits T={int(logits.size(1))} targets T={int(targets.size(1))}")

    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=ignore_index)
    return loss