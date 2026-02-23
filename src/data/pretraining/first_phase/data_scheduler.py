import numpy as np
from typing import Dict

class DataScheduler:
    def __init__(self, steps: int, overall_ratios: Dict[str, float], end_ratios: Dict[str, float]):
        self.steps = steps
        self.overall_ratios = overall_weights
        self.end_ratioss = end_weights
        self.current_params = {}

    def _calculate_params(self):
        params = {}
        for key in self.overall_ratios:
            W = self.overall_ratios[key]
            Pe = self.end_ratios[key]
            S = self.steps

            m1 = (W - Pe*S + 0.5*Pe*(S-St)) / ( (St**2 / 2) + St*(S-St) - (k*(S-St)**2 / 2) )


    def get_ratios(self, step: int) -> Dict[str, float]:
        ratios = {}
        for key, r in self.current_params.items():
            val = r["ps"] + r["m1"] * step
            ratios[key] = max(0.005, val)
        
        total = sum(ratios.values())
        for key in ratios:
            ratios[key] /= total
        
        return ratios
