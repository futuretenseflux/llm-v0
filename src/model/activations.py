#activations.py
import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, dim_in, intermediate_size):
        super().__init__()
        self.dim_in = dim_in
        self.fc = nn.Linear(dim_in, intermediate_size * 2, bias=False)
        self.swish = nn.SiLU()

    def forward(self, x):
        out = self.fc(x)
        gate, val = out.chunk(2, dim=-1)
        return self.swish(gate) * val