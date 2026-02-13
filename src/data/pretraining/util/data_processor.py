import yaml
from pathlib import Path
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer
import array
from typing import Union, List, Optional


def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent.parent.parent / "configs" / "lm.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def process_tokenize_save(dataset: Dataset, output_prefix: str, output_dir: str, tokenizer_name: str = "facebook/galactica-6.7b", buffer_size: int = 100_000_000):
    """
    Process, tokenize and save a dataset to binary files.
    
    Args:
        dataset: The dataset to process
        output_prefix: Prefix for output binary files (e.g., 'books', 'code')
        output_dir: Directory to save the processed binary files
        tokenizer_name: Name of the tokenizer to use
        buffer_size: Number of tokens to buffer before writing to file
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    shard = 0
    buffer = array.array('H')

    for example in dataset:
        if len(buffer) >= buffer_size:
            output_path = os.path.join(output_dir, f"{output_prefix}_{shard}.bin")
            with open(output_path, "wb") as f:
                buffer.tofile(f)
            buffer = array.array('H')
            shard += 1

        from .normalize import clean_scientific_text
        normalized = clean_scientific_text(example['text'])
        tokenized_ids = tokenizer.encode(normalized, add_special_tokens=False)
        buffer.extend(tokenized_ids)
        buffer.append(tokenizer.eos_token_id)

    if buffer:
        output_path = os.path.join(output_dir, f"{output_prefix}_{shard}.bin")
        with open(output_path, "wb") as f:
            buffer.tofile(f)


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
    """
    Load dataset(s), optionally concatenate, shuffle, and process.
    
    Args:
        dataset_key: Key for the dataset in config (e.g., 'books', 'code')
        output_prefix: Prefix for output binary files (defaults to dataset_key)
        output_dir: Directory to save processed files (from config if not provided)
        split: Dataset split to use
        shuffle_seed: Seed for shuffling
        subset: Optional subset name for the dataset
        concatenate_datasets_list: List of additional dataset subsets to concatenate
        tokenizer_name: Name of the tokenizer to use (from config if not provided)
    """
    config = load_config()
    
    # Get dataset name from config
    dataset_name = config['datasets'][dataset_key]
    
    # Use config values if not provided
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
    process_tokenize_save(dataset, output_prefix, output_dir, tokenizer_name)
