#decoder.py
import torch
import torch.nn as nn
from llm.model.attention import GroupedQueryAttention
from llm.model.activations import SwiGLU

class DecoderLayer(nn.Module):
    def __init__(self, dim_model, dim_k, num_q_heads, group_size, intermediate_size, eps=1e-6, dropout=0.1):
        super().__init__()
        self.gq_attn = GroupedQueryAttention(num_q_heads, group_size, dim_model, dim_k, dropout=dropout, enable_rope=True)
        self.rms_norm_1 = nn.RMSNorm(normalized_shape=dim_model, eps=eps)
        self.attention_dropout = nn.Dropout(p=dropout)
        self.swiglu = SwiGLU(dim_in=dim_model, intermediate_size=intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, dim_model, bias=False)
        self.ffn_dropout = nn.Dropout(p=dropout)
        self.rms_norm_2 = nn.RMSNorm(normalized_shape=dim_model, eps=eps)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)

    def forward(self, x):
        batch_size, seq_length = x.shape[:2]
        mask = torch.tril(torch.ones(seq_length, seq_length, device=x.device)).bool()
        mask = mask[None, None, None, :, :]
        norm1 = self.rms_norm_1(x)
        context = self.gq_attn(norm1, mask=mask)
        context = self.attention_dropout(context)
        x = context + x
        norm2 = self.rms_norm_2(x)
        act = self.swiglu(norm2)
        ffn_out = self.down_proj(act)
        ffn_dropout = self.ffn_dropout(ffn_out)
        x = ffn_dropout + x
        return x