"""Proposal interface and learned-draft implementation."""
from dataclasses import dataclass
from typing import Protocol

import torch

from storage.paged_cache import PagedCache
from runtime.sampling import draw
from model.tokenizer import check_tokenizers


@dataclass
class Proposal:
    tokens: list[int]
    distributions: list[torch.Tensor | None]

    def __post_init__(self):
        if len(self.tokens) != len(self.distributions):
            raise ValueError("each proposed token needs a proposal distribution")


@dataclass(frozen=True)
class Commit:
    sequence: int
    base_length: int
    accepted: int
    emitted: int | None
    stopped: bool


class Proposer(Protocol):
    calls: int

    def prepare(self, prompts, max_new_tokens, params, rngs): ...
    def propose(self, indices, histories, budgets, eos_token_id=None): ...
    def commit(self, commits): ...
    def close(self): ...


class DraftModelProposer:
    """Autoregressive proposals from a smaller decoder model."""

    def __init__(self, target, draft, pool=None):
        if target.device != draft.device or target.vocab_size != draft.vocab_size:
            raise ValueError("target and draft must share device and token vocabulary")
        check_tokenizers(target, draft)
        if target.training or draft.training:
            raise ValueError("target and draft must be in eval mode")
        self.draft = draft
        self.pool = pool
        self.caches = []
        self.next_logits = {}
        self.params = None
        self.rngs = None
        self.calls = 0

    def _call(self, indices, rows):
        chunks = [torch.tensor([row], device=self.draft.device) for row in rows]
        caches = [self.caches[i] for i in indices]
        if hasattr(self.draft, "forward_batch"):
            outputs = self.draft.forward_batch(chunks, caches, last_only=True)
        else:
            outputs = [self.draft(chunk, caches=cache)[:, -1:] for chunk, cache in zip(chunks, caches)]
        self.calls += 1
        return {i: output[0] for i, output in zip(indices, outputs)}

    def prepare(self, prompts, max_new_tokens, params, rngs):
        self.params = params
        self.rngs = rngs
        self.calls = 0
        self.caches = []
        for prompt in prompts:
            capacity = len(prompt) + max_new_tokens
            if capacity > self.draft.config.max_seq_len:
                raise ValueError("generation budget exceeds draft context")
            cache = PagedCache(self.pool) if self.pool is not None else self.draft.make_kv_caches(1, capacity)
            self.caches.append(cache)
        indices = list(range(len(prompts)))
        outputs = self._call(indices, prompts)
        self.next_logits = {i: outputs[i][-1] for i in indices}

    def propose(self, indices, histories, budgets, eos_token_id=None):
        proposals = {i: Proposal([], []) for i in indices}
        for step in range(max(budgets.values(), default=0)):
            eligible = [i for i in indices if step < budgets[i] and
                        (not proposals[i].tokens or proposals[i].tokens[-1] != eos_token_id)]
            if not eligible:
                break
            rows = []
            for index in eligible:
                if self.params.temperature == 0:
                    distribution = None
                    token = int(self.next_logits[index].argmax().item())
                else:
                    distribution = self.params.probabilities(self.next_logits[index])
                    token = draw(distribution, self.rngs[index])
                proposals[index].tokens.append(token)
                proposals[index].distributions.append(distribution)
                rows.append([token])
            outputs = self._call(eligible, rows)
            for index in eligible:
                self.next_logits[index] = outputs[index][-1]
        return proposals

    def commit(self, commits):
        continuing, rows = [], []
        for commit in commits:
            self.draft.rollback(self.caches[commit.sequence], commit.base_length + commit.accepted)
            if not commit.stopped and commit.emitted is not None:
                continuing.append(commit.sequence)
                rows.append([commit.emitted])
        if continuing:
            outputs = self._call(continuing, rows)
            for index in continuing:
                self.next_logits[index] = outputs[index][-1]

    def close(self):
        for cache in self.caches:
            if isinstance(cache, PagedCache):
                cache.close()
        self.caches = []
