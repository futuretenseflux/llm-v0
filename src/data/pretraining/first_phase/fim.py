import random
from typing import Optional

import torch
from torch.utils.data import Dataset


class FIMDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        seq_length: int,
        fim_prefix_id: int,
        fim_middle_id: int,
        fim_suffix_id: int,
        fim_prob: float = 0.5,
        rng_seed: Optional[int] = None,
    ):
        self.base = base
        self.seq_length = int(seq_length)
        self.fim_prefix_id = int(fim_prefix_id)
        self.fim_middle_id = int(fim_middle_id)
        self.fim_suffix_id = int(fim_suffix_id)
        self.fim_prob = float(fim_prob)
        self._rng = random.Random(rng_seed)

        if not (0.0 <= self.fim_prob <= 1.0):
            raise ValueError("fim_prob must be in [0, 1]")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        input_ids, targets = self.base[idx]

        if self.fim_prob <= 0.0 or (self.fim_prob < 1.0 and self._rng.random() >= self.fim_prob):
            return input_ids, targets

        if input_ids.numel() != self.seq_length or targets.numel() != self.seq_length:
            return input_ids, targets

        # Reconstruct the (seq_length + 1) token window so we can re-shift after FIM.
        seq = torch.cat([input_ids, targets[-1:]], dim=0)

        # We need to insert 3 control tokens (<PRE>, <SUF>, <MID>) while keeping total length.
        # To do that, we drop 3 tokens from the end of the original window.
        if seq.numel() < 4:
            return input_ids, targets

        content = seq[:-3]
        n = int(content.numel())
        if n < 2:
            return input_ids, targets

        # Pick two cut points to split into prefix / middle / suffix.
        a = self._rng.randint(0, n - 2)
        b = self._rng.randint(a + 1, n - 1)

        prefix = content[:a]
        middle = content[a:b]
        suffix = content[b:]

        fim_seq = torch.cat(
            [
                torch.tensor([self.fim_prefix_id], dtype=seq.dtype),
                prefix,
                torch.tensor([self.fim_suffix_id], dtype=seq.dtype),
                suffix,
                torch.tensor([self.fim_middle_id], dtype=seq.dtype),
                middle,
            ],
            dim=0,
        )

        # Sanity: ensure we preserved the original window length.
        # If something goes wrong, fall back to original sample.
        if fim_seq.numel() != seq.numel():
            return input_ids, targets

        new_input = fim_seq[:-1]
        new_targets = fim_seq[1:]
        return new_input, new_targets
