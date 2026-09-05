import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MHA(nn.Module):
    def __init__(self, d_model, n_head, causal=True):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head, self.d_head = n_head, d_model // n_head
        self.causal = causal

        self.Wqkv = nn.Linear(d_model, 3 * d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, d_model = x.shape

        qkv = self.Wqkv(x)  # (B, T, 3 * d_model)
        q, k, v = qkv.chunk(3, dim=-1) # (B, T, d_model)

        q = torch.reshape(q, (B, T, self.n_head, self.d_head)).transpose(1, 2) # (B, h, T, d_h)
        k = torch.reshape(k, (B, T, self.n_head, self.d_head)).transpose(1, 2) # (B, h, T, d_h)
        v = torch.reshape(v, (B, T, self.n_head, self.d_head)).transpose(1, 2) # (B, h, T, d_h)

        attn_score = q @ k.transpose(-2, -1) # (B, h, T, d_h) @ (B, h, d_h, T) -> (B, h, T, T)
        attn_logit = attn_score / math.sqrt(self.d_head) # (B, h, T, T)
        if self.causal:
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
            attn_logit = attn_logit.masked_fill(mask, float("-inf"))
        attn_logit = attn_logit - attn_logit.amax(dim=-1, keepdim=True) # (B, h, T, T)
        exp_logit = torch.exp(attn_logit) # (B, h, T, T)
        attn_weight = exp_logit / exp_logit.sum(dim=-1, keepdim=True) # (B, h, T, T)
        attn_output = attn_weight @ v # (B, h, T, T) @ (B, h, T, d_h) -> (B, h, T, d_h)

        attn_output = attn_output.transpose(1, 2).contiguous() # (B, T, h, d_h)

        o = torch.reshape(attn_output, (B, T, d_model))
        return self.Wo(o)

