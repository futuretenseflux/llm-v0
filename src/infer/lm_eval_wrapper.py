
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Tuple, Optional, Union

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance

from src.infer.inference import load_inference_bundle

@register_model("local_model", "my_custom_model")
class MyCustomLM(LM):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 1,
        max_length: int = 2048,
        **kwargs
    ):
        super().__init__()
        self.model_path = model_path
        self._device = device
        self.batch_size_per_gpu = int(batch_size)
        self.max_length = int(max_length)

        long_context = bool(kwargs.pop("long_context", True))
        
        # Load the model and tokenizer using your existing infrastructure
        self.model, self.tokenizer, self.device = load_inference_bundle(
            model_path=model_path,
            device=device,
            long_context=long_context,
        )
        
        # Ensure tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Enforce right padding for loglikelihood which assumes [tokens, pad] structure
        self.tokenizer.padding_side = "right"

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood of generating a continuation given a context.
        """
        results = []
        
        for i in tqdm(range(0, len(requests), self.batch_size_per_gpu), desc="Evaluating loglikelihood"):
            batch = requests[i : i + self.batch_size_per_gpu]
            
            # Extract inputs
            contexts = [req.args[0] for req in batch]
            continuations = [req.args[1] for req in batch]
            
            # Combine context + continuation
            full_texts = [ctx + cont for ctx, cont in zip(contexts, continuations)]
            
            # Tokenize
            # Note: We must use right-padding because the model uses causal masking (is_causal=True)
            # and does not accept an explicit attention mask to ignore left-padded tokens.
            encodings = self.tokenizer(
                full_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                add_special_tokens=False
            ).to(self.device)
            
            input_ids = encodings.input_ids
            attention_mask = encodings.attention_mask
            
            with torch.no_grad():
                # Forward pass
                logits = self.model(input_ids)
                
                # Compute logprobs
                # shift logits and labels
                # logits: [B, L, V] -> [B, L-1, V]
                # labels: [B, L] -> [B, L-1]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                # Flatten to compute loss/logprobs
                # but we need to keep batch structure to sum properly
                log_probs = F.log_softmax(shift_logits, dim=-1)
                
                # Gather logprobs for the actual tokens
                # [B, L-1]
                token_log_probs = torch.gather(
                    log_probs, 
                    dim=-1, 
                    index=shift_labels.unsqueeze(-1)
                ).squeeze(-1)

            # Post-process per request in the batch
            for j, (ctx, cont) in enumerate(zip(contexts, continuations)):
                # We need to mask out:
                # 1. The context part (we only care about probability of continuation)
                # 2. The padding parts
                
                # Re-tokenize context to find its length
                # This is slightly inefficient but safe. 
                # Alternatively, we could tokenize context separately and concat ids.
                ctx_ids = self.tokenizer(ctx, add_special_tokens=False).input_ids
                ctx_len = len(ctx_ids)
                
                # Total length of the sequence (excluding padding)
                # attention_mask[j] has 1s for valid tokens
                seq_len = attention_mask[j].sum().item()
                
                # The continuation starts at ctx_len
                # However, since we shifted labels, the label at index k corresponds to prediction for token k+1.
                # We want probability of tokens in continuation.
                # Context: [t0, t1, ... t_{m-1}]
                # Full:    [t0, t1, ... t_{m-1}, c0, c1, ... c_{n-1}]
                # Labels:      [t1, ... t_{m-1}, c0, c1, ... c_{n-1}] (shifted)
                # Indices:     0        m-2      m-1
                
                # The prediction for c0 comes from t_{m-1} (which is at index m-1 in original, m-1 in shifted?? No)
                # Original indices: 0, 1, ..., m-1 (last context char), m (first cont char)
                # Shifted logits at index i predict token at i+1.
                # We want predictions for c0, c1, ...
                # c0 is at index m in original input_ids.
                # So we want shifted logit at index m-1 (which predicts m).
                
                start_idx = ctx_len - 1
                if start_idx < 0:
                    start_idx = 0
                    
                # End index is the end of the valid sequence - 1 (because shifted)
                end_idx = seq_len - 1
                
                # Extract relevant logprobs
                cont_log_probs = token_log_probs[j, start_idx:end_idx]
                
                # Sum logprobs
                total_log_prob = cont_log_probs.sum().item()
                
                # Check greedy decoding
                # Get the argmax tokens for the continuation region
                greedy_indices = shift_logits[j, start_idx:end_idx].argmax(dim=-1)
                target_indices = shift_labels[j, start_idx:end_idx]
                is_greedy = (greedy_indices == target_indices).all().item()
                
                results.append((total_log_prob, is_greedy))
                
        return results

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        """
        Compute log-likelihood of the entire string (perplexity evaluation).
        """
        # Simplistic implementation: just score the whole string
        # For true rolling, one might use sliding windows, but this is a start.
        results = []
        for i in tqdm(range(0, len(requests), self.batch_size_per_gpu), desc="Evaluating loglikelihood_rolling"):
            batch = requests[i : i + self.batch_size_per_gpu]
            texts = [req.args[0] for req in batch]
            
            encodings = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                add_special_tokens=False
            ).to(self.device)
            
            input_ids = encodings.input_ids
            attention_mask = encodings.attention_mask
            
            with torch.no_grad():
                logits = self.model(input_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            
            for j in range(len(batch)):
                seq_len = attention_mask[j].sum().item()
                # Sum all logprobs for the valid sequence
                # (excluding the first token which isn't predicted)
                total_log_prob = token_log_probs[j, :seq_len-1].sum().item()
                results.append(total_log_prob)
                
        return results

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Generate text until a stop sequence.
        """
        results = []
        
        for req in tqdm(requests, desc="Generating"):
            context = req.args[0]
            gen_kwargs = req.args[1]
            
            until = gen_kwargs.get("until", [])
            max_gen_toks = gen_kwargs.get("max_gen_toks", 256)
            temperature = gen_kwargs.get("temperature", 0.0)
            top_p = gen_kwargs.get("top_p", None)
            
            if isinstance(until, str):
                until = [until]
                
            input_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)
            
            # Keep track of generated tokens
            generated_tokens = []
            decoded_text = ""
            
            with torch.no_grad():
                for _ in range(max_gen_toks):
                    logits = self.model(input_ids)
                    next_token_logits = logits[:, -1, :]
                    
                    if temperature == 0.0:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    else:
                        scaled_logits = next_token_logits / float(temperature)
                        probs = torch.softmax(scaled_logits, dim=-1)
                        
                        if top_p is not None and 0.0 < top_p < 1.0:
                            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                            
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = False
                            
                            sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)
                            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                            
                            next_token_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
                            next_token = sorted_indices.gather(-1, next_token_sorted_idx)
                        else:
                            next_token = torch.multinomial(probs, num_samples=1)
                            
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    token_id = next_token.item()
                    generated_tokens.append(token_id)
                    
                    # Check for stop conditions
                    # We decode incrementally or fully. Fully is safer for multi-token stop sequences.
                    decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    stop_found = False
                    for stop_seq in until:
                        if stop_seq in decoded_text:
                            # Truncate
                            decoded_text = decoded_text.split(stop_seq)[0]
                            stop_found = True
                            break
                    
                    if stop_found:
                        break
                        
            results.append(decoded_text)
            
        return results
