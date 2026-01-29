#attention.py
import torch
import torch.nn as nn
import math
from llm.model.pe import RoPE

class GroupedQueryAttention(nn.Module):
    def __init__(self, num_q_heads, group_size, dim_model, dim_k, dropout=0.1, enable_rope=True):
        super().__init__()
        assert dim_model % num_q_heads == 0, "dim_model must be divisible by num_q_heads"
        assert num_q_heads % group_size == 0, "num_q_heads must be divisible by group_size"
        self.enable_rope = enable_rope
        self.group_size = group_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_q_heads // group_size
        self.dim_model = dim_model
        self.dim_k = dim_k
        self.q_norm = nn.RMSNorm(self.dim_k)
        self.k_norm = nn.RMSNorm(self.dim_k)
        self.rope = None
        if self.enable_rope:
            self.rope = RoPE()
        self.q_proj = nn.Linear(dim_model, self.dim_k * self.num_q_heads, bias=False)
        self.k_proj = nn.Linear(dim_model, self.dim_k * self.num_kv_heads, bias=False)
        self.v_proj = nn.Linear(dim_model, self.dim_k * self.num_kv_heads, bias=False)
        self.fc = nn.Linear(self.dim_k * num_q_heads, dim_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_length = x.shape[:2]
        q = self.q_proj(x).view(batch_size, seq_length, self.num_q_heads, self.dim_k).transpose(1,2)
        k = self.k_proj(x).view(batch_size, seq_length, self.num_kv_heads, self.dim_k).transpose(1,2)
        v = self.v_proj(x).view(batch_size, seq_length, self.num_kv_heads, self.dim_k).transpose(1,2)
        if self.enable_rope:
            q = self.rope.apply(q)
            k = self.rope.apply(k)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(batch_size, self.num_kv_heads, self.group_size, seq_length, self.dim_k)
        k = k.transpose(-1,-2).unsqueeze(2)
        scores = torch.matmul(q,k) / math.sqrt(self.dim_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = nn.functional.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        v = v.unsqueeze(2)
        out = torch.matmul(attn_weights, v)
        out = out.contiguous().view(batch_size, self.num_q_heads, seq_length, self.dim_k)
        out = self.fc(out.transpose(1,2).contiguous().view(batch_size, seq_length, self.num_q_heads * self.dim_k))
        return out