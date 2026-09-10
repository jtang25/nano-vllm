from pathlib import Path
import torch

from model.config import ModelConfig
from model.layers import DecoderLM
from model.tokenizer import ByteTokenizer


def save_checkpoint(model, path, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = getattr(model, "tokenizer_spec", ByteTokenizer.name)
    payload = {"format_version": 2 if isinstance(tokenizer, dict) else 1, "config": model.config.to_dict(), "tokenizer": tokenizer,
               "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
               "metadata": metadata or {}}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu", dtype=torch.float32):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    tokenizer = payload.get("tokenizer")
    if not ((payload.get("format_version") == 1 and tokenizer == ByteTokenizer.name) or
            (payload.get("format_version") == 2 and isinstance(tokenizer, dict) and tokenizer.get("type") == "huggingface")):
        raise ValueError("unsupported checkpoint format or tokenizer")
    # Avoid initializing a second full-size fp32 model during every benchmark load.
    with torch.device("meta"):
        model = DecoderLM(config=ModelConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"], strict=True, assign=True)
    if model.config.tie_embeddings:
        model.lm_head.weight = model.token_embedding.weight
    model.tokenizer_spec = tokenizer
    model.checkpoint_metadata = payload.get("metadata", {})
    for layer in model.layers:
        head_dim = layer.attention.d_head
        layer.attention.inv_freq = model.config.rope_base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    return model.to(device=device, dtype=dtype).eval()
