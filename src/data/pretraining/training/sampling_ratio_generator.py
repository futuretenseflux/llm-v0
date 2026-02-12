import numpy as np
import math

DATASET_ORDER = ["web", "books", "papers", "code", "math", "convo", "primer"]

def get_sampling_ratios(current_token_count):
    total_budget = 20.0
    phase_tokens = np.array([1.00, 12.10, 0.15, 6.75])
    phase_ends = np.cumsum(phase_tokens)
    
    r = np.array([
        [0.083, 0.719, 0.100, 0.051, 0.020], # Web
        [0.083, 0.040, 0.404, 0.051, 0.020], # Books
        [0.083, 0.040, 0.100, 0.255, 0.350], # Papers
        [0.083, 0.040, 0.150, 0.204, 0.250], # Code
        [0.083, 0.040, 0.200, 0.357, 0.350], # Math
        [0.083, 0.120, 0.050, 0.082, 0.010], # Convo
        [0.500, 0.001, 0.000, 0.000, 0.000]  # Primer
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