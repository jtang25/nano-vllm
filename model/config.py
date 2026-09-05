from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 259
    d_model: int = 128
    n_layers: int = 4
    n_q_heads: int = 4
    n_kv_heads: int = 2
    d_ff: int = 384
    max_seq_len: int = 2048
    norm_eps: float = 1e-6
    rope_base: float = 10000.0
    n_experts: int = 0
    experts_per_token: int = 2
    tie_embeddings: bool = False
    attention_backend: str = "reference"
    qkv_bias: bool = True
    output_bias: bool = True
    mlp_bias: bool = True

    def __post_init__(self):
        for name in ("vocab_size", "d_model", "n_layers", "n_q_heads", "n_kv_heads", "d_ff", "max_seq_len"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.n_q_heads or self.n_q_heads % self.n_kv_heads:
            raise ValueError("invalid GQA head geometry")
        if self.head_dim % 2:
            raise ValueError("RoPE needs an even head dimension")
        if not all(math.isfinite(x) and x > 0 for x in (self.norm_eps, self.rope_base)):
            raise ValueError("norm_eps and rope_base must be finite and positive")
        if type(self.n_experts) is not int or self.n_experts < 0:
            raise ValueError("n_experts must be a nonnegative integer")
        if self.n_experts and not 1 <= self.experts_per_token <= self.n_experts:
            raise ValueError("invalid experts_per_token")
        if self.attention_backend not in ("reference", "sdpa"):
            raise ValueError("attention_backend must be reference or sdpa")

    @property
    def head_dim(self):
        return self.d_model // self.n_q_heads

    def to_dict(self):
        return asdict(self)
