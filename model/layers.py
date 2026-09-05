import torch
from torch import nn
from torch.nn import functional as F

from model.attention import GQA
from storage.cache import KVCache
from model.config import ModelConfig
from storage.paged_cache import PagedCache


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        work = x if x.dtype == torch.float64 else x.float()
        variance = work.square().mean(dim=-1, keepdim=True)
        return (work * torch.rsqrt(variance + self.eps)).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, bias=True):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    """Top-k routing, normalized selected weights, no token dropping."""
    def __init__(self, d_model, d_ff, n_experts, experts_per_token, bias=True):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(d_model, d_ff, bias) for _ in range(n_experts))
        self.experts_per_token = experts_per_token

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        logits = self.router(flat).float()
        selected_logits, indices = logits.topk(self.experts_per_token, dim=-1)
        weights = selected_logits.softmax(-1).to(x.dtype)
        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            rows, slots = torch.where(indices == expert_id)
            if rows.numel():
                contribution = expert(flat[rows]) * weights[rows, slots, None]
                output.index_add_(0, rows, contribution)
        return output.reshape_as(x)

    def balance_loss(self, x):
        """Optional training loss: encourage tokens to use the expert pool."""
        probabilities = self.router(x.reshape(-1, x.shape[-1])).float().softmax(-1)
        selected = probabilities.topk(self.experts_per_token, dim=-1).indices
        usage = F.one_hot(selected, len(self.experts)).float().mean((0, 1))
        return len(self.experts) * (usage * probabilities.mean(0)).sum()


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_q_heads, n_kv_heads, d_ff, norm_eps=1e-6,
                 rope_base=10000.0, n_experts=0, experts_per_token=2, backend="reference",
                 qkv_bias=True, output_bias=True, mlp_bias=True):
        super().__init__()
        self.attention_norm = RMSNorm(d_model, norm_eps)
        self.attention = GQA(d_model, n_q_heads, n_kv_heads, rope_base=rope_base, backend=backend, qkv_bias=qkv_bias, output_bias=output_bias)
        self.ffn_norm = RMSNorm(d_model, norm_eps)
        self.ffn = MoE(d_model, d_ff, n_experts, experts_per_token, mlp_bias) if n_experts else SwiGLU(d_model, d_ff, mlp_bias)

    def forward(self, x, positions, cache=None):
        x = x + self.attention(self.attention_norm(x), positions, cache)
        return x + self.ffn(self.ffn_norm(x))

    def forward_packed(self, x, positions, lengths, caches):
        x = x + self.attention.forward_packed(self.attention_norm(x), positions, lengths, caches)
        return x + self.ffn(self.ffn_norm(x))


class DecoderLM(nn.Module):
    def __init__(self, vocab_size=259, d_model=128, n_layers=4, n_q_heads=4, n_kv_heads=2,
                 d_ff=384, norm_eps=1e-6, tie_embeddings=False, *, config=None):
        super().__init__()
        self.config = config or ModelConfig(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
            n_q_heads=n_q_heads, n_kv_heads=n_kv_heads, d_ff=d_ff, norm_eps=norm_eps, tie_embeddings=tie_embeddings)
        c = self.config
        self.vocab_size, self.d_model, self.n_layers = c.vocab_size, c.d_model, c.n_layers
        self.n_kv_heads, self.d_head = c.n_kv_heads, c.head_dim
        self.token_embedding = nn.Embedding(c.vocab_size, c.d_model)
        self.layers = nn.ModuleList(DecoderBlock(c.d_model, c.n_q_heads, c.n_kv_heads, c.d_ff,
            c.norm_eps, c.rope_base, c.n_experts, c.experts_per_token, c.attention_backend,
            c.qkv_bias, c.output_bias, c.mlp_bias) for _ in range(c.n_layers))
        self.final_norm = RMSNorm(c.d_model, c.norm_eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        if c.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @property
    def device(self):
        return self.token_embedding.weight.device

    @property
    def dtype(self):
        return self.token_embedding.weight.dtype

    def make_kv_caches(self, batch_size, capacity):
        if not 0 < capacity <= self.config.max_seq_len:
            raise ValueError("invalid cache capacity")
        return [KVCache(batch_size, self.n_kv_heads, capacity, self.d_head, device=self.device, dtype=self.dtype)
                for _ in self.layers]

    def _validate_ids(self, ids):
        if ids.ndim != 2 or ids.shape[0] == 0 or ids.shape[1] == 0 or ids.dtype != torch.long:
            raise ValueError("input_ids must be nonempty [B,T] int64")
        if ids.device != self.device:
            raise ValueError("input device differs from model")
        if bool(((ids < 0) | (ids >= self.vocab_size)).any()):
            raise ValueError("token ID outside vocabulary")

    def _prepare_cache(self, caches, batch, count):
        if self.training or torch.is_grad_enabled():
            raise ValueError("cached execution requires eval() and inference_mode()")
        layers = caches.layers if isinstance(caches, PagedCache) else caches
        if len(layers) != self.n_layers or len({c.length for c in layers}) != 1:
            raise ValueError("one synchronized cache is required per layer")
        start = layers[0].length
        if start + count > self.config.max_seq_len:
            raise ValueError("model context exceeded")
        if isinstance(caches, PagedCache):
            pool = caches.pool
            expected = (self.n_layers, self.n_kv_heads, self.d_head)
            actual = (pool.k.shape[0], pool.k.shape[2], pool.k.shape[-1])
            if batch != 1 or actual != expected or pool.k.device != self.device or pool.k.dtype != self.dtype:
                raise ValueError("paged pool geometry/device/dtype mismatch")
            caches.reserve(start + count)
        else:
            for cache in layers:
                if cache.length + count > cache.capacity:
                    raise ValueError("cache capacity exceeded")
                if cache.k.shape[:2] != (batch, self.n_kv_heads) or cache.k.shape[-1] != self.d_head:
                    raise ValueError("cache shape mismatch")
                if cache.k.device != self.device or cache.k.dtype != self.dtype:
                    raise ValueError("cache device/dtype mismatch")
        return layers, start

    @staticmethod
    def rollback(caches, start):
        if isinstance(caches, PagedCache):
            caches.truncate(start)
        else:
            for cache in caches:
                cache.truncate(start)

    def forward(self, input_ids, positions=None, caches=None, last_only=False):
        """[B,T] -> [B,T,V]. Cached inputs contain only the new tokens."""
        self._validate_ids(input_ids)
        b, t = input_ids.shape
        if t > self.config.max_seq_len:
            raise ValueError("model context exceeded")
        layer_caches, start = ([None] * self.n_layers, 0)
        if caches is not None:
            layer_caches, start = self._prepare_cache(caches, b, t)
        expected = torch.arange(start, start + t, device=self.device)
        try:
            if positions is None:
                positions = expected
            if positions.shape != (t,) or positions.device != self.device or positions.dtype != torch.long:
                raise ValueError("positions must be int64 [T] on the model device")
            if caches is not None and not torch.equal(positions, expected):
                raise ValueError("positions must continue from cache length")
            x = self.token_embedding(input_ids)
            for layer, cache in zip(self.layers, layer_caches):
                x = layer(x, positions, cache)
            return self.lm_head(self.final_norm(x[:, -1:] if last_only else x))
        except Exception:
            if caches is not None:
                self.rollback(caches, start)
            raise

    def forward_batch(self, chunks, caches, last_only=False):
        """Packed continuous batching: shared projection/MLP calls, isolated attention."""
        if not chunks or len(chunks) != len(caches) or len({id(c) for c in caches}) != len(caches):
            raise ValueError("provide distinct caches for a nonempty batch")
        prepared = []
        try:
            for ids, cache in zip(chunks, caches):
                self._validate_ids(ids)
                if ids.shape[0] != 1:
                    raise ValueError("each packed chunk must be [1,T]")
                layers, start = self._prepare_cache(cache, 1, ids.shape[1])
                prepared.append((layers, start))
            lengths = [ids.shape[1] for ids in chunks]
            positions = [torch.arange(start, start + size, device=self.device) for (_, start), size in zip(prepared, lengths)]
            x = self.token_embedding(torch.cat(chunks, dim=1))
            for i, layer in enumerate(self.layers):
                x = layer.forward_packed(x, positions, lengths, [layers[i] for layers, _ in prepared])
            if last_only:
                indices = torch.tensor(lengths, device=x.device).cumsum(0) - 1
                logits = self.lm_head(self.final_norm(x[:, indices]))
                return list(logits.split(1, dim=1))
            logits = self.lm_head(self.final_norm(x))
            return list(logits.split(lengths, dim=1))
        except Exception:
            for cache, (_, start) in zip(caches, prepared):
                self.rollback(cache, start)
            raise
