import numpy as np

def final_curriculum():
    total_budget = 20.0
    # Calculated optimal lengths
    phase_tokens = np.array([1.00, 12.10, 0.15, 6.75])
    
    # Transition Ratios (Rows: Web, Books, Papers, Code, Math, Convo, Primer)
    # Columns: t0, t1, t2, t3, t4
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
        tokens = 0
        for p in range(4):
            avg = (r[i, p] + r[i, p+1]) / 2
            tokens += avg * phase_tokens[p]
        print(f"{name:<10} | {tokens:<12.2f} | {tokens/total_budget:.2%}")

final_curriculum()