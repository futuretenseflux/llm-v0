import os
import sys
import yaml
import torch
import numpy as np
from collections import Counter

# Add the project root to the path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.pretraining.training.dataset import BinaryTokenDataset

def analyze_sample(input_ids):
    """Analyze a single sequence of tokens."""
    tokens = input_ids.numpy()
    
    # Check for all zeros
    zeros = np.sum(tokens == 0)
    zero_ratio = zeros / len(tokens)
    
    # Check for constant values
    unique_tokens = np.unique(tokens)
    num_unique = len(unique_tokens)
    
    # Check for repetition (token[i] == token[i-1])
    if len(tokens) > 1:
        diffs = tokens[1:] - tokens[:-1]
        repeats = np.sum(diffs == 0)
        repeat_ratio = repeats / (len(tokens) - 1)
    else:
        repeat_ratio = 0.0
        
    return {
        "zero_ratio": zero_ratio,
        "num_unique": num_unique,
        "repeat_ratio": repeat_ratio,
        "min": np.min(tokens),
        "max": np.max(tokens),
        "sample_tokens": tokens[:20].tolist()
    }

def main():
    config_path = "configs/lm.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    output_dir = config["output_dir"]
    seq_length = config["seq_length"]
    stride = config.get("stride", seq_length)
    
    print(f"Checking data in: {output_dir}")
    print(f"Sequence Length: {seq_length}")
    
    datasets_to_check = ["books", "code", "conv_forum", "math", "papers", "primer", "web"]
    
    found_issues = False
    
    for name in datasets_to_check:
        print(f"\n=== Checking Dataset: {name} ===")
        try:
            # Check if files exist first
            import glob
            files = glob.glob(os.path.join(output_dir, f"{name}_*.bin"))
            if not files:
                print(f"  [WARN] No .bin files found for {name} in {output_dir}")
                continue
                
            dataset = BinaryTokenDataset(output_dir, name, seq_length, stride)
            if len(dataset) == 0:
                print("  [WARN] Dataset is empty (0 samples).")
                continue
                
            print(f"  Total samples: {len(dataset)}")
            
            # Check random samples
            num_check = min(10, len(dataset))
            indices = np.linspace(0, len(dataset)-1, num_check, dtype=int)
            
            for i in indices:
                input_ids, targets = dataset[i]
                stats = analyze_sample(input_ids)
                
                print(f"  Sample {i}:")
                print(f"    Tokens (first 20): {stats['sample_tokens']}...")
                print(f"    Unique tokens: {stats['num_unique']} / {seq_length}")
                print(f"    Zero ratio: {stats['zero_ratio']:.2%} {'[CRITICAL: MOSTLY ZEROS]' if stats['zero_ratio'] > 0.9 else ''}")
                print(f"    Repeat ratio: {stats['repeat_ratio']:.2%} {'[CRITICAL: HIGH REPETITION]' if stats['repeat_ratio'] > 0.5 else ''}")
                
                if stats['zero_ratio'] > 0.5 or stats['repeat_ratio'] > 0.5 or stats['num_unique'] < 5:
                    found_issues = True
                    print("    [!] SUSPICIOUS DATA DETECTED")
                    
        except Exception as e:
            print(f"  [ERROR] Failed to load {name}: {e}")
            found_issues = True

    print("\n=== Summary ===")
    if found_issues:
        print("POSSIBLE DATA FAULT DETECTED. See logs above.")
    else:
        print("Data statistics look nominal (no obvious all-zero or constant files).")

if __name__ == "__main__":
    main()
