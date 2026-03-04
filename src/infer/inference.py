from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoTokenizer

from src.model.transformer import Transformer
from src.utils.config import load_lm_config


_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"

_INFERENCE_BUNDLE_CACHE: Dict[Tuple[str, str, bool], Tuple[Transformer, Any, str]] = {}

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


def build_chatml_prompt(
    messages: Sequence[Dict[str, str]],
    *,
    reasoning_on: bool = False,
) -> str:
    """Build a ChatML-like prompt matching the SFT data format.

    Notes:
    - Messages are serialized as:
      <|im_start|>{role}\n{content}<|im_end|>\n
    - The returned prompt always ends with an *open* assistant turn:
      <|im_start|>assistant\n
      so the model can generate the assistant content.
    - If reasoning_on is True, appends "[INTERNAL_REASONING=ON]" to the system
      prompt content.
    """

    if len(messages) == 0:
        raise ValueError("messages must be non-empty")

    parts: List[str] = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported role: {role}")

        if role == "system" and reasoning_on:
            if len(content) == 0:
                content = "[INTERNAL_REASONING=ON]"
            else:
                content = f"{content}\n[INTERNAL_REASONING=ON]"

        parts.append(f"{_IM_START}{role}\n{content}{_IM_END}")

    # Open assistant turn for generation.
    parts.append(f"{_IM_START}assistant\n")
    return "\n".join(parts)


def _endswith_tokens(seq: torch.Tensor, suffix: torch.Tensor) -> bool:
    if seq.numel() < suffix.numel():
        return False
    return bool(torch.equal(seq[-suffix.numel() :], suffix))


def run_chat_inference(
    model_path: str,
    messages: Sequence[Dict[str, str]],
    *,
    reasoning_on: bool = False,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    device: Optional[str] = None,
    long_context: bool = True,
) -> str:
    """Run inference on a post-SFT model with ChatML formatting.

    Returns only the assistant completion (not the full prompt).
    """

    prompt = build_chatml_prompt(messages, reasoning_on=reasoning_on)
    model, tokenizer, device = load_inference_bundle(model_path, device=device, long_context=bool(long_context))

    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    prompt_len = int(input_ids.shape[1])

    stop_ids = tokenizer(_IM_END, add_special_tokens=False).input_ids
    if len(stop_ids) == 0:
        raise RuntimeError("Failed to tokenize stop string <|im_end|>.")
    stop_ids_t = torch.tensor(stop_ids, dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]

            if temperature is None or float(temperature) == 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                scaled_logits = next_token_logits / float(temperature)
                probs = torch.softmax(scaled_logits, dim=-1)

                if top_p is not None and 0.0 < float(top_p) < 1.0:
                    next_token = _sample_top_p(probs, float(top_p))
                else:
                    next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop when the generated sequence ends with <|im_end|>
            gen_seq = input_ids[0, prompt_len:]
            if _endswith_tokens(gen_seq, stop_ids_t):
                break

    gen_ids = input_ids[0, prompt_len:]
    if _endswith_tokens(gen_ids, stop_ids_t):
        gen_ids = gen_ids[: -stop_ids_t.numel()]

    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def load_inference_bundle(
    model_path: str,
    *,
    device: Optional[str] = None,
    long_context: Optional[bool] = None,
) -> Tuple[Transformer, Any, str]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_lm_config()
    if long_context is None:
        long_context = bool(config.get("long_context", True))

    cache_key = (model_path, device, bool(long_context))
    cached = _INFERENCE_BUNDLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

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
        long_context=bool(long_context),
    )

    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state_dict dict, got: {type(state_dict)}")

    # Handle checkpoints saved from torch.compile(model) (prefix: '_orig_mod.')
    # and DDP-wrapped models (prefix: 'module.').
    def _strip_prefix(sd: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        if all(k.startswith(prefix) for k in sd.keys()):
            return {k[len(prefix) :]: v for k, v in sd.items()}
        return sd

    state_dict = _strip_prefix(state_dict, "_orig_mod.")
    state_dict = _strip_prefix(state_dict, "module.")

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
    long_context: Optional[bool] = None,
) -> str:
    model, tokenizer, device = load_inference_bundle(model_path, device=device, long_context=long_context)

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


def score_candidates_loglikelihood(
    model_path: str,
    prompt: str,
    candidates: list[str],
    *,
    device: Optional[str] = None,
    long_context: Optional[bool] = None,
) -> list[float]:
    model, tokenizer, device = load_inference_bundle(model_path, device=device, long_context=long_context)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if len(candidates) == 0:
        return []

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if len(prompt_ids) == 0:
        raise ValueError("Prompt must tokenize to at least one token.")

    texts = [prompt + c for c in candidates]
    batch = tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_ids)
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        targets = input_ids[:, 1:]
        token_logps = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        cand_start = len(prompt_ids) - 1
        if cand_start < 0:
            cand_start = 0

        pos_idx = torch.arange(token_logps.shape[1], device=device).unsqueeze(0)
        valid_positions = attention_mask[:, 1:].bool()
        cand_positions = pos_idx >= cand_start
        mask = valid_positions & cand_positions

        scores = (token_logps * mask).sum(dim=1)
        return scores.detach().float().cpu().tolist()