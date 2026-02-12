import torch
from torch.utils.data import Dataset
import numpy as np

class BinaryTokenDataset(Dataset):
    def __init__(self, data_dir: str, dataset_name: str, seq_length: int, stride=None, token_dtype=np.uint16):
        self.data = np.memmap(f"{data_dir}/{dataset_name}.bin", dtype=token_dtype, mode="r")
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
        return torch.from_numpy(self.data[start:end]).long()

        
    