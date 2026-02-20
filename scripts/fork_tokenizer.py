import argparse
from pathlib import Path

from transformers import AutoTokenizer


def _ceil_to_multiple(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="facebook/galactica-6.7b")
    parser.add_argument("--out", type=str, default="tokenizers/galactica-6.7b-fork")
    parser.add_argument("--pad_to_multiple", type=int, default=128)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)

    # Note: The Galactica tokenizer already defines the canonical special-token IDs:
    #   0=<s> (BOS), 1=<pad> (PAD), 2=</s> (EOS), 3=<unk> (UNK).
    # This script only appends new tokens, so these base IDs remain unchanged.

    if tokenizer.bos_token is None:
        tokenizer.bos_token = "<s>"
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"
    if tokenizer.unk_token is None:
        tokenizer.unk_token = "<unk>"

    reserved_control_tokens = [
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|think|>",
        "<|/think|>",
    ]

    additional_special_tokens = list(tokenizer.additional_special_tokens)
    for t in reserved_control_tokens:
        if t not in additional_special_tokens:
            additional_special_tokens.append(t)

    tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

    target_vocab_size = _ceil_to_multiple(len(tokenizer), int(args.pad_to_multiple))
    if target_vocab_size > len(tokenizer):
        extra = target_vocab_size - len(tokenizer)
        padding_tokens = [f"<|reserved_special_token_{i}|>" for i in range(extra)]
        additional_special_tokens = list(tokenizer.additional_special_tokens)
        additional_special_tokens.extend(padding_tokens)
        tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

    tokenizer.save_pretrained(str(out_dir))

    print(f"Saved tokenizer to: {out_dir}")
    print(f"Base: {args.base}")
    print(f"Final vocab size (len(tokenizer)): {len(tokenizer)}")
    print("Special tokens:")
    print(f"  bos={tokenizer.bos_token} id={tokenizer.bos_token_id}")
    print(f"  pad={tokenizer.pad_token} id={tokenizer.pad_token_id}")
    print(f"  eos={tokenizer.eos_token} id={tokenizer.eos_token_id}")
    print(f"  unk={tokenizer.unk_token} id={tokenizer.unk_token_id}")


if __name__ == "__main__":
    main()
