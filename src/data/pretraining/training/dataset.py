import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os

class BinaryTokenDataset(Dataset):
    def __init__(self, data_dir: str, dataset_name: str, seq_length: int, stride=None, token_dtype=np.uint16):
        files = sorted(glob.glob(os.path.join(data_dir, f"{dataset_name}_*.bin")))
        self.data = np.concatenate([
            np.memmap(f, dtype=token_dtype, mode="r")
            for f in files
        ])
        self.seq_length = seq_length
        self.stride = stride if stride is not None else seq_length
    
    def __len__(self):
        n = len(self.data)
        if n <= self.seq_length:
            return 0
        return (n - self.seq_length - 1) // self.stride + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_length
        input_ids = torch.from_numpy(self.data[start:end]).long()
        targets = torch.from_numpy(self.data[start+1:end+1]).long()  # Shifted by 1
        return input_ids, targets

        
    