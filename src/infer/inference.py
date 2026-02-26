from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer

from src.model.transformer import Transformer
from src.utils.config import load_lm_config

_INFERENCE_BUNDLE_CACHE: Dict[Tuple[str, str], Tuple[Transformer, Any, str]] = {}

def _sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    next_token_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_indices.gather(-1, next_token_sorted_idx)
    return next_token


def load_inference_bundle(
    model_path: str,
    *,
    device: Optional[str] = None,
) -> Tuple[Transformer, Any, str]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_key = (model_path, device)
    cached = _INFERENCE_BUNDLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    config = load_lm_config()
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_model"], use_fast=True)

    model = Transformer(
        vocab_size=int(config["vocab_size"]),
        dim_model=int(config["dim_model"]),
        dim_k=int(config["dim_k"]),
        num_q_heads=int(config["num_q_heads"]),
        group_size=int(config["group_size"]),
        num_decoder_layers=int(config["num_decoder_layers"]),
        intermediate_size=int(config["intermediate_size"]),
        eps=float(config["eps"]),
        dropout=float(config["dropout"]),
        long_context=bool(config.get("long_context", False)),
    )

    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)

    infer_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = model.to(device=device, dtype=infer_dtype)
    model.eval()

    bundle = (model, tokenizer, device)
    _INFERENCE_BUNDLE_CACHE[cache_key] = bundle
    return bundle


def run_inference(
    model_path: str,
    input_text: str,
    *,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    device: Optional[str] = None,
) -> str:
    model, tokenizer, device = load_inference_bundle(model_path, device=device)

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]

            if temperature is None or temperature == 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                scaled_logits = next_token_logits / float(temperature)
                probs = torch.softmax(scaled_logits, dim=-1)

                if top_p is not None and 0.0 < top_p < 1.0:
                    next_token = _sample_top_p(probs, float(top_p))
                else:
                    next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return output_text