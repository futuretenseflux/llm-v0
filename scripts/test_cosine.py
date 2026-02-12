import numpy as np

def final_curriculum_cosine():
    total_budget = 20.0
    phase_tokens = np.array([1.00, 12.10, 0.15, 6.75])
    
    # Transition Ratios (same as your original matrix)
    r = np.array([
        [0.083, 0.719, 0.100, 0.051, 0.020], # Web
        [0.083, 0.040, 0.404, 0.051, 0.020], # Books
        [0.083, 0.040, 0.100, 0.255, 0.350], # Papers
        [0.083, 0.040, 0.150, 0.204, 0.250], # Code
        [0.083, 0.040, 0.200, 0.357, 0.350], # Math
        [0.083, 0.120, 0.050, 0.082, 0.010], # Convo
        [0.500, 0.001, 0.000, 0.000, 0.000]  # Primer
    ])

    sources = ["Web", "Books", "Papers", "Code", "Math", "Convo", "Primer"]
    
    print(f"{'Source':<10} | {'Tokens (B)':<12} | {'Total %'}")
    print("-" * 35)

    for i, name in enumerate(sources):
        total_source_tokens = 0
        for p in range(4):
            w_start = r[i, p]
            w_end = r[i, p+1]
            
            # The integral of [w_end + 0.5*(w_start - w_end)*(1 + cos(x*pi))] from 0 to 1
            # simplifies back to (w_start + w_end) / 2 because the cosine term 
            # integrates to 0 over a full half-cycle!
            
            # However, if you use a "Warm Restart" or non-symmetric annealing, 
            # the distribution changes. For standard cosine over the phase:
            phase_avg = (w_start + w_end) / 2 
            
            total_source_tokens += phase_avg * phase_tokens[p]
            
        print(f"{name:<10} | {total_source_tokens:<12.2f} | {total_source_tokens/total_budget:.2%}")

final_curriculum_cosine()