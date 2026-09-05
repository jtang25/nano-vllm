import math

import torch
from torch import nn
from torch.nn import functional as F


class GQA(nn.Module):
    def __init__(self, d_model, n_q_heads, n_kv_heads, causal=True, rope_base=10000.0, backend="reference", qkv_bias=True, output_bias=True):
        super().__init__()
        if min(d_model, n_q_heads, n_kv_heads) <= 0 or d_model % n_q_heads or n_q_heads % n_kv_heads:
            raise ValueError("invalid GQA dimensions")
        if not math.isfinite(rope_base) or rope_base <= 0:
            raise ValueError("rope_base must be finite and positive")
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_q_heads
        self.group_size = n_q_heads // n_kv_heads
        self.causal = causal
        self.rope_base = rope_base
        self.backend = backend
        if self.d_head % 2 or backend not in ("reference", "sdpa"):
            raise ValueError("invalid RoPE dimension or attention backend")
        self.Wq = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.Wkv = nn.Linear(d_model, 2 * n_kv_heads * self.d_head, bias=qkv_bias)
        self.Wo = nn.Linear(d_model, d_model, bias=output_bias)
        inv_freq = rope_base ** (-torch.arange(0, self.d_head, 2, dtype=torch.float32) / self.d_head)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rope(self, x, cos, sin):
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def _attention(self, q, k, v, positions, cache):
        b, t, _ = q.shape
        q = q.reshape(b, t, self.n_q_heads, self.d_head).transpose(1, 2)
        k = k.reshape(b, t, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.reshape(b, t, self.n_kv_heads, self.d_head).transpose(1, 2)
        # Rebuild in fp32 after model.half(): frequency precision matters at long positions.
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        channels = torch.arange(0, self.d_head, 2, device=q.device, dtype=dtype)
        angles = positions.to(dtype)[:, None] * self.rope_base ** (-channels / self.d_head)
        cos, sin = angles.cos()[None, None], angles.sin()[None, None]
        q = self._apply_rope(q.to(dtype), cos, sin).to(v.dtype)
        k = self._apply_rope(k.to(dtype), cos, sin).to(v.dtype)
        if cache is not None:
            if not self.causal or torch.is_grad_enabled():
                raise ValueError("cached attention requires causal inference with gradients disabled")
            expected = torch.arange(cache.length, cache.length + t, device=q.device)
            if not torch.equal(positions, expected):
                raise ValueError("positions must continue from cache.length")
            if hasattr(cache, "attend"):
                fresh_prefill = cache.length == 0 and t > 1 and self.backend == "sdpa"
                cache.append(k, v)
                if fresh_prefill:
                    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
                    return out.transpose(1, 2).reshape(b, t, -1)
                out = cache.attend(q, positions)
                return out.transpose(1, 2).reshape(b, t, -1)
            k, v = cache.append(k, v)
            key_positions = torch.arange(k.shape[2], device=q.device)
        else:
            key_positions = positions
        visible = key_positions[None, :] <= positions[:, None]
        if self.backend == "sdpa":
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=visible if self.causal else None, enable_gqa=True)
        else:
            # A group dimension expresses KV sharing without storing repeated heads.
            query = q.reshape(b, self.n_kv_heads, self.group_size, t, self.d_head).to(dtype)
            scores = query @ k[:, :, None].to(dtype).transpose(-2, -1) / math.sqrt(self.d_head)
            if self.causal:
                scores = scores.masked_fill(~visible, -torch.inf)
            out = (scores.softmax(-1) @ v[:, :, None].to(dtype)).reshape(b, self.n_q_heads, t, self.d_head).to(v.dtype)
        return out.transpose(1, 2).reshape(b, t, -1)

    def forward(self, x, positions, cache=None):
        if x.ndim != 3 or x.shape[1] == 0 or positions.shape != (x.shape[1],):
            raise ValueError("expected x [B,T,D] and positions [T]")
        if positions.device != x.device or positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("positions must be integer indices on the input device")
        k, v = self.Wkv(x).chunk(2, dim=-1)
        return self.Wo(self._attention(self.Wq(x), k, v, positions, cache))

    def forward_packed(self, x, positions, lengths, caches):
        """Project all requests together; isolate each request's attention history."""
        q = self.Wq(x)
        k, v = self.Wkv(x).chunk(2, dim=-1)
        outputs = []
        start = 0
        for size, pos, cache in zip(lengths, positions, caches):
            end = start + size
            outputs.append(self._attention(q[:, start:end], k[:, start:end], v[:, start:end], pos, cache))
            start = end
        return self.Wo(torch.cat(outputs, dim=1))
