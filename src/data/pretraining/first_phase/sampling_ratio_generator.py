import numpy as np
import math

DATASET_ORDER = ["web", "books", "papers", "code", "math", "conv_forum", "primer"]

def get_sampling_ratios(tokens_elapsed):
    current_token_count = tokens_elapsed / 1e9
    total_budget = 20.0
    phase_tokens = np.array([0.40, 9.85, 6.25, 3.50])

    phase_ends = np.cumsum(phase_tokens)

    r = np.array([
        [0.0833, 0.62, 0.25, 0.08, 0.02], # Web
        [0.0833, 0.06, 0.35, 0.05, 0.02], # Books
        [0.0833, 0.05, 0.10, 0.25, 0.35], # Papers
        [0.0833, 0.05, 0.15, 0.18, 0.20], # Code
        [0.0833, 0.05, 0.15, 0.35, 0.40], # Math
        [0.0833, 0.10, 0.03, 0.09, 0.01], # Convo
        [0.5000, 0.00, 0.00, 0.00, 0.00], # Primer
    ])
    sources = DATASET_ORDER

    phase_idx = np.searchsorted(phase_ends, current_token_count)
    if phase_idx >= len(phase_tokens): # Training complete
        return {s: r[i, -1] for i, s in enumerate(sources)}

    phase_start = phase_ends[phase_idx - 1] if phase_idx > 0 else 0
    phase_len = phase_tokens[phase_idx]
    progress = (current_token_count - phase_start) / phase_len

    cos_mult = 0.5 * (1 - math.cos(math.pi * progress))

    ratios = {}
    for i, name in enumerate(sources):
        w_start = r[i, phase_idx]
        w_end = r[i, phase_idx + 1]
        
        current_weight = w_start + (w_end - w_start) * cos_mult
        ratios[name] = round(current_weight, 4)

    return ratios