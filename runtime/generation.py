from dataclasses import dataclass
import torch

from storage.paged_cache import PagedCache
from runtime.sampling import SamplingParams, draw


@dataclass
class GenerationResult:
    tokens: list[int]
    generated: list[int]
    forward_calls: int
    cached_tokens: int


@torch.inference_mode()
def generate(model, prompt, max_new_tokens=32, params=None, eos_token_id=None, seed=0, pool=None):
    """Last emitted token is pending: output length = cache length + 1."""
    params = params or SamplingParams()
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    ids = torch.as_tensor(prompt, dtype=torch.long, device=model.device).reshape(1, -1)
    model._validate_ids(ids)
    if ids.shape[1] + max_new_tokens > model.config.max_seq_len:
        raise ValueError("generation budget exceeds context")
    if max_new_tokens == 0:
        return GenerationResult(ids[0].tolist(), [], 0, 0)
    cache = PagedCache(pool) if pool is not None else model.make_kv_caches(1, ids.shape[1] + max_new_tokens)
    generator = torch.Generator(device=model.device).manual_seed(seed)
    tokens, generated, calls = ids[0].tolist(), [], 0
    try:
        for _ in range(max_new_tokens):
            logits = model(ids, caches=cache, last_only=True)
            calls += 1
            token = draw(params.probabilities(logits[0, -1]), generator)
            tokens.append(token)
            generated.append(token)
            if token == eos_token_id:
                break
            ids = torch.tensor([[token]], device=model.device)
        length = cache.length if isinstance(cache, PagedCache) else cache[0].length
        return GenerationResult(tokens, generated, calls, length)
    finally:
        if isinstance(cache, PagedCache):
            cache.close()
