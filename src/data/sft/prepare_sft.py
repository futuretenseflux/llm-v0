import os
import re
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
import multiprocessing

from src.utils.config import load_lm_config
from src.data.pretraining.util.normalize import clean_scientific_text

tokenizer = None

def process_example(examples):
    global tokenizer
    config = load_lm_config()
    if tokenizer is None:
        tokenizer_path = config['tokenizer_model']
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except OSError:
            tokenizer = AutoTokenizer.from_pretrained("tokenizers/galactica-6.7b-fork")

    max_length = config['sft_max_length']

    texts = [clean_scientific_text(t) for t in examples['text']]

    encodings = tokenizer(
        texts,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    input_ids_batch = encodings['input_ids']
    offsets_batch = encodings['offset_mapping']

    pattern = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"
    labels_batch = []
    for text, input_ids, offsets in zip(texts, input_ids_batch, offsets_batch):
        labels = [-100] * len(input_ids)

        for match in re.finditer(pattern, text, re.DOTALL):
            start_char, end_char = match.span(1)
            end_token_start_char, end_token_end_char = match.end(1), match.end(0)

            for i, (tok_start, tok_end) in enumerate(offsets):
                if tok_start >= start_char and tok_end <= end_char:
                    labels[i] = input_ids[i]
                if tok_start >= end_token_start_char and tok_end <= end_token_end_char:
                    labels[i] = input_ids[i]

        labels_batch.append(labels)

    return {
        "input_ids": input_ids_batch,
        "labels": labels_batch,
    }

def prepare_sft():
    config = load_lm_config()
    
    dataset_name = config['datasets']['sft']
    tokenizer_path = config['tokenizer_model']
    output_dir = os.path.join(config['data_output_dir'], 'sft')
    
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split='train')

    ds = ds.shuffle(seed=config.get('shuffle_seed', 42))
    
    print(f"Loading tokenizer: {tokenizer_path}")
    try:
        global tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except OSError:
        print(f"Tokenizer not found at {tokenizer_path}. Please ensure you have run scripts/fork_tokenizer.py")
        return

    print("Processing dataset...")
    num_proc = config.get('num_proc', multiprocessing.cpu_count())
    
    processed_ds = ds.map(
        process_example,
        batched=True,
        batch_size=config.get('map_batch_size', 128),
        num_proc=num_proc,
        remove_columns=ds.column_names,
        desc="Normalizing and tokenizing"
    )
    
    # Save the processed dataset
    print(f"Saving processed dataset to {output_dir}")
    processed_ds.save_to_disk(output_dir)
    print("Done.")

if __name__ == "__main__":
    prepare_sft()
