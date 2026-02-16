import yaml
from pathlib import Path
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer
import array
from typing import Union, List, Optional
import os
from .normalize import clean_scientific_text
import numpy as np

_tokenizer = None

def load_config():
    config_path = Path(__file__).parent.parent.parent.parent.parent / "configs" / "lm.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def tokenize_batch(batch, tokenizer_name):
    global _tokenizer

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    texts = [clean_scientific_text(t) for t in batch["text"]]
    tokens = _tokenizer(texts, add_special_tokens=False)

    eos = _tokenizer.eos_token_id or 2

    return {
        "ids": [ids + [eos] for ids in tokens["input_ids"]]
    }

def tokenize_dataset(dataset, tokenizer_name):
    dataset = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=1000,
        num_proc=max(1, int(os.cpu_count() * 0.8)),
        fn_kwargs={"tokenizer_name": tokenizer_name},
        remove_columns=dataset.column_names,
    )
    return dataset

def write_tokenized_dataset(
    dataset,
    output_dir,
    output_prefix: str,
    shard_size_tokens=500_000_000,
    dtype=np.uint16,
):
    os.makedirs(output_dir, exist_ok=True)

    shard = 0
    token_count = 0

    output_path = os.path.join(output_dir, f"{output_prefix}_{shard}.bin")
    f = open(output_path, "wb", buffering=1024*1024*64)  # 64MB buffer

    try:
        for example in dataset:
            arr = np.asarray(example["ids"], dtype=dtype)
            arr.tofile(f)
            token_count += arr.size

            if token_count >= shard_size_tokens:
                f.close()
                shard += 1
                token_count = 0

                output_path = os.path.join(output_dir, f"{output_prefix}_{shard}.bin")
                f = open(output_path, "wb", buffering=1024*1024*64)

    finally:
        f.close()


# def write_tokenized_dataset(dataset, output_dir, output_prefix: str, shard_size_tokens=500_000_000, dtype=np.uint16):
#     os.makedirs(output_dir, exist_ok=True)

#     shard = 0
#     token_buffer = []
#     token_count = 0

#     for example in dataset:
#         ids = example["ids"]
#         token_buffer.extend(ids)
#         token_count += len(ids)

#         if token_count >= shard_size_tokens:
#             write_shard(token_buffer, output_prefix, output_dir, shard, dtype)
#             shard  += 1
#             token_buffer = []
#             token_count = 0
        
#     if token_buffer:
#         write_shard(token_buffer, output_prefix, output_dir, shard, dtype)

# def write_shard(buffers, prefix, output_dir, shard_id, dtype):
#     output_path = os.path.join(output_dir, f"{prefix}_{shard_id}.bin")

#     arr = np.asarray(buffers, dtype=dtype)

#     with open(output_path, "wb") as f:
#         arr.tofile(f)


def load_and_process_dataset(
    dataset_key: str,
    output_prefix: Optional[str] = None,
    output_dir: Optional[str] = None,
    split: str = "train",
    shuffle_seed: int = 42,
    subset: Optional[str] = None,
    concatenate_datasets_list: Optional[List[str]] = None,
    tokenizer_name: Optional[str] = None
):
    config = load_config()
    
    dataset_name = config['datasets'][dataset_key]
    
    if tokenizer_name is None:
        tokenizer_name = config.get('tokenizer_model', 'facebook/galactica-6.7b')
    if output_prefix is None:
        output_prefix = dataset_key
    if output_dir is None:
        output_dir = config.get('output_dir', 'data/processed')
    
    if concatenate_datasets_list:
        datasets = []
        for subset_name in concatenate_datasets_list:
            ds = load_dataset(dataset_name, subset_name, split=split)
            datasets.append(ds)
        dataset = concatenate_datasets(datasets)
    else:
        dataset = load_dataset(dataset_name, subset, split=split) if subset else load_dataset(dataset_name, split=split)
    
    dataset = dataset.shuffle(seed=shuffle_seed)
    tokenized_normalized_suffixed = tokenize_dataset(dataset, tokenizer_name)
    write_tokenized_dataset(tokenized_normalized_suffixed, output_dir, output_prefix, shard_size_tokens=500_000_000, dtype=np.uint16)
