import os
from typing import Dict, Optional, Sequence, Tuple, Union
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from src.utils.config import load_lm_config


class SFTDiskDataset(Dataset):
    def __init__(self, dataset_path: str, return_dict: bool = False):
        self.dataset_path = dataset_path
        self.ds = load_from_disk(dataset_path)
        self.return_dict = bool(return_dict)

    def __len__(self) -> int:
        return int(len(self.ds))

    def __getitem__(self, idx: int):
        ex = self.ds[int(idx)]

        input_ids_list = ex["input_ids"]
        labels_list = ex["labels"]

        input_ids = torch.tensor(input_ids_list[:-1], dtype=torch.long)
        targets = torch.tensor(labels_list[1:], dtype=torch.long)

        if self.return_dict:
            return {"input_ids": input_ids, "targets": targets}
        return input_ids, targets

def sft_collate_fn(
    batch: Sequence[Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]],
    *,
    pad_token_id: int,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(batch) == 0:
        raise ValueError("Empty batch")

    if isinstance(batch[0], dict):
        input_ids_list = [b["input_ids"] for b in batch]  # type: ignore[index]
        targets_list = [b["targets"] for b in batch]  # type: ignore[index]
    else:
        input_ids_list = [b[0] for b in batch]  # type: ignore[index]
        targets_list = [b[1] for b in batch]  # type: ignore[index]

    max_len = max(int(x.numel()) for x in input_ids_list)

    input_ids = torch.full((len(batch), max_len), int(pad_token_id), dtype=torch.long)
    targets = torch.full((len(batch), max_len), int(ignore_index), dtype=torch.long)

    for i, (x, y) in enumerate(zip(input_ids_list, targets_list)):
        n = int(x.numel())
        input_ids[i, :n] = x
        targets[i, :n] = y

    return input_ids, targets

def build_sft_dataset(
    *,
    dataset_path: Optional[str] = None,
    config: Optional[dict] = None,
    return_dict: bool = False,
) -> SFTDiskDataset:
    if dataset_path is None:
        if config is None:
            config = load_lm_config()
        dataset_path = os.path.join(config["data_output_dir"], "sft")
    return SFTDiskDataset(dataset_path, return_dict=return_dict)
