#pe.py
import torch

class RoPE:
    def __init__(self, base=50000):
        self.cos_cache = None
        self.sin_cache = None
        self.theta = None
        self.cached_seq = 0
        self.cached_d = 0
        self.base = base

    def get_rot_cached(self, d, seq_length, device):
        if self.cos_cache is None:
            self.theta = self.base ** (-2 * torch.arange(d//2, dtype=torch.float32, device=device) / d)
            ms = torch.arange(seq_length, device=device)
            angles = torch.einsum('i,j->ij', ms, self.theta)
            self.cos_cache = torch.cos(angles)
            self.sin_cache = torch.sin(angles)
            self.cached_d = d
            self.cached_seq = seq_length
            needed_dhalf = d // 2
            return self.cos_cache[:seq_length, :needed_dhalf], self.sin_cache[:seq_length, :needed_dhalf]

        needed_dhalf = d // 2
        if d != self.cached_d:
            self.theta = self.base ** (-2 * torch.arange(needed_dhalf, dtype=torch.float32, device=device) / d)
            ms = torch.arange(self.cached_seq, device=device)
            angles = torch.einsum('i,j->ij', ms, self.theta)
            self.cos_cache = torch.cos(angles)
            self.sin_cache = torch.sin(angles)
            self.cached_d = d

        if seq_length > self.cached_seq:
            new_ms = torch.arange(self.cached_seq, seq_length, dtype=torch.float32, device=device)
            new_angles = torch.einsum('i,j->ij', new_ms, self.theta)
            new_cos = torch.cos(new_angles)
            new_sin = torch.sin(new_angles)
            self.cos_cache = torch.cat([self.cos_cache, new_cos], dim=0)
            self.sin_cache = torch.cat([self.sin_cache, new_sin], dim=0)
            self.cached_seq = seq_length

        return self.cos_cache[:seq_length, :needed_dhalf], self.sin_cache[:seq_length, :needed_dhalf]

    def apply(self, t):
        seq_length, d = t.shape[-2:]
        r_cos, r_sin = self.get_rot_cached(d, seq_length, t.device)
        t_even = t[..., 0::2]
        t_odd = t[..., 1::2]
        t_conj = torch.empty_like(t)
        t_conj[..., 0::2] = -t_odd
        t_conj[..., 1::2] = t_even
        return t * r_cos.repeat_interleave(2, dim=-1) + t_conj * r_sin.repeat_interleave(2, dim=-1)