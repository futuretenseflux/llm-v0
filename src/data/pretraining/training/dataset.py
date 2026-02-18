import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os

class BinaryTokenDataset(Dataset):
    def __init__(self, data_dir: str, dataset_name: str, seq_length: int, stride=None, token_dtype=np.uint16):
        files = sorted(glob.glob(os.path.join(data_dir, f"{dataset_name}_*.bin")))
        self._mmaps = [np.memmap(f, dtype=token_dtype, mode="r") for f in files]
        lengths = [len(m) for m in self._mmaps]
        self._offsets = np.cumsum([0] + lengths, dtype=np.int64)
        self.seq_length = seq_length
        self.stride = stride if stride is not None else seq_length
    
    def __len__(self):
        n = int(self._offsets[-1])
        if n <= self.seq_length:
            return 0
        return (n - self.seq_length - 1) // self.stride + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_length

        input_np = self._slice(start, end)
        target_np = self._slice(start + 1, end + 1)

        if input_np.size != self.seq_length or target_np.size != self.seq_length:
            raise ValueError(
                f"Expected slices of length {self.seq_length}, got input={int(input_np.size)} target={int(target_np.size)} "
                f"(idx={int(idx)} start={int(start)} end={int(end)})"
            )

        input_ids = torch.from_numpy(np.array(input_np, copy=True)).long()
        targets = torch.from_numpy(np.array(target_np, copy=True)).long()  # Shifted by 1
        return input_ids, targets

    def _slice(self, start: int, end: int) -> np.ndarray:
        if start < 0 or end < start or end > int(self._offsets[-1]):
            raise IndexError("Slice out of range")
        if start == end:
            return np.empty((0,), dtype=self._mmaps[0].dtype)

        parts = []
        pos = start
        while pos < end:
            file_idx = int(np.searchsorted(self._offsets, pos, side="right") - 1)
            local_start = pos - int(self._offsets[file_idx])
            take = min(end - pos, len(self._mmaps[file_idx]) - local_start)
            parts.append(self._mmaps[file_idx][local_start:local_start + take])
            pos += take

        if len(parts) == 1:
            return np.asarray(parts[0])
        return np.concatenate([np.asarray(p) for p in parts])
    