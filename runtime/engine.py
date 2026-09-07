"""Continuous batching with chunked prefill and conservative admission credits."""
from collections import deque
from dataclasses import dataclass, field
import math
import time
import torch

from storage.paged_cache import PagedCache
from storage.prefix_cache import PrefixStore
from runtime.sampling import SamplingParams, draw


@dataclass
class Request:
    id: str
    tokens: list[int]
    prompt_length: int
    max_new_tokens: int
    params: SamplingParams
    eos_token_id: int | None
    rng: torch.Generator
    credit: int
    cache: PagedCache | None = None
    generated: list[int] = field(default_factory=list)
    status: str = "waiting"
    submitted_at: float = field(default_factory=time.perf_counter)
    first_token_at: float | None = None
    finished_at: float | None = None
    token_times: list[float] = field(default_factory=list)


class Engine:
    def __init__(self, model, pool, max_batch_tokens=128, max_requests=8, *, chunked_prefill=True,
                 decode_first=False, prefix_entries=0):
        if model.training:
            raise ValueError("engine requires model.eval()")
        if min(max_batch_tokens, max_requests) < 1:
            raise ValueError("scheduler budgets must be positive")
        self.model, self.pool = model, pool
        self.max_batch_tokens, self.max_requests = max_batch_tokens, max_requests
        self.requests = {}
        self.waiting = deque()
        self.active = deque()
        self.credits = 0
        self.forward_calls = 0
        self.chunked_prefill = chunked_prefill
        self.decode_first = decode_first
        self.prefixes = PrefixStore(prefix_entries)

    def submit(self, prompt, max_new_tokens=32, params=None, eos_token_id=None, seed=0):
        tokens = list(prompt)
        if not tokens or any(type(t) is not int or not 0 <= t < self.model.vocab_size for t in tokens):
            raise ValueError("prompt must contain valid integer token IDs")
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if len(tokens) + max_new_tokens > self.model.config.max_seq_len:
            raise ValueError("request exceeds model context")
        credit = math.ceil((len(tokens) + max_new_tokens) / self.pool.block_size)
        if credit > self.pool.n_blocks:
            raise ValueError("request cannot fit in the block pool")
        if eos_token_id is not None and not 0 <= eos_token_id < self.model.vocab_size:
            raise ValueError("invalid EOS token ID")
        params = params or SamplingParams()
        if params.top_k is not None and params.top_k > self.model.vocab_size:
            raise ValueError("top_k exceeds vocabulary")
        request_id = str(len(self.requests))
        rng = torch.Generator(device=self.model.device).manual_seed(seed)
        request = Request(request_id, tokens, len(tokens), max_new_tokens, params or SamplingParams(),
                          eos_token_id, rng, credit)
        self.requests[request_id] = request
        if max_new_tokens == 0:
            request.status, request.finished_at = "finished", time.perf_counter()
        else:
            self.waiting.append(request)
        return request_id

    def _finish(self, request, status):
        if request.status in ("finished", "cancelled", "failed"):
            return
        request.status = status
        request.finished_at = time.perf_counter()
        if request.cache is not None:
            request.cache.close()
            self.credits -= request.credit
        if request in self.active:
            self.active.remove(request)

    def cancel(self, request_id):
        request = self.requests[request_id]
        if request.status == "waiting":
            self.waiting.remove(request)
        if request.status in ("waiting", "running"):
            self._finish(request, "cancelled")

    @torch.inference_mode()
    def step(self):
        while self.waiting and len(self.active) < self.max_requests:
            request = self.waiting[0]
            if self.credits + request.credit + self.prefixes.pinned_blocks > self.pool.n_blocks:
                self.prefixes.clear()
            if self.credits + request.credit > self.pool.n_blocks:
                break
            self.waiting.popleft()
            request.cache = self.prefixes.get(request.tokens) or PagedCache(self.pool)
            request.status = "running"
            self.credits += request.credit
            self.active.append(request)
        if not self.active:
            return []
        if self.credits + self.prefixes.pinned_blocks > self.pool.n_blocks:
            self.prefixes.clear()
        selected, chunks, budget = [], [], self.max_batch_tokens
        # Round-robin rotation prevents a long prefill from starving other work.
        candidates = list(self.active)
        if self.decode_first:
            candidates.sort(key=lambda r: not bool(r.generated))
        for request in candidates:
            if budget == 0:
                break
            start = request.cache.length
            size = min(len(request.tokens) - start, budget)
            if not self.chunked_prefill and not request.generated:
                remaining = len(request.tokens) - start
                if selected and remaining > budget:
                    continue
                size = remaining
            chunks.append(torch.tensor([request.tokens[start:start + size]], device=self.model.device))
            selected.append(request)
            budget = max(0, budget - size)
        try:
            logits = self.model.forward_batch(chunks, [r.cache for r in selected], last_only=True)
            self.forward_calls += 1
            emitted = []
            for request, output in zip(selected, logits):
                if request.cache.length == len(request.tokens):
                    if not request.generated:
                        self.prefixes.put(request.tokens[:-1], request.cache)
                    token = draw(request.params.probabilities(output[0, -1]), request.rng)
                    request.tokens.append(token)
                    request.generated.append(token)
                    request.token_times.append(time.perf_counter())
                    if request.first_token_at is None:
                        request.first_token_at = request.token_times[-1]
                    emitted.append((request.id, token))
                    if token == request.eos_token_id or len(request.generated) == request.max_new_tokens:
                        self._finish(request, "finished")
            self.active.rotate(-len(selected))
            return emitted
        except Exception:
            for request in selected:
                self._finish(request, "failed")
            raise

    def run(self):
        while self.waiting or self.active:
            self.step()
        return self.requests

    def close(self):
        for request in list(self.requests.values()):
            self.cancel(request.id)
        self.prefixes.clear()
