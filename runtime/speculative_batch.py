"""Shared batched verifier for learned-draft and lookup proposals."""
from dataclasses import dataclass, field
import time

import torch

from storage.paged_cache import PagedCache
from runtime.proposers import Commit, DraftModelProposer
from runtime.sampling import SamplingParams, draw, residual_distribution


@dataclass
class BatchSpecResult:
    tokens: list[list[int]]
    generated: list[list[int]]
    token_times: list[list[float]]
    prefill_done_at: float | None = None
    target_calls: int = 0
    draft_calls: int = 0
    proposer_calls: int = 0
    proposed: int = 0
    accepted: int = 0
    bonus_tokens: int = 0
    rounds: list[dict] = field(default_factory=list)


def _accept_or_correct(logits, token, proposal_distribution, params, rng):
    if params.temperature == 0:
        target_token = int(logits.argmax().item())
        return token == target_token, None if token == target_token else target_token
    target_distribution = params.probabilities(logits)
    if proposal_distribution is None:
        probability = target_distribution[token]
        residual = target_distribution.clone()
        residual[token] = 0
        residual /= residual.sum()
    else:
        probability = torch.minimum(torch.ones_like(target_distribution[token]),
                                    target_distribution[token] / proposal_distribution[token])
        residual = None
    accepted = bool(torch.rand((), device=logits.device, generator=rng) < probability)
    if accepted:
        return True, None
    if residual is None:
        residual = residual_distribution(target_distribution, proposal_distribution)
    return False, draw(residual, rng)


@torch.inference_mode()
def speculative_batch_with_proposer(target, proposer, prompts, max_new_tokens=32, lookahead=4,
                                    params=None, seed=0, target_pool=None, eos_token_id=None):
    if not prompts or type(max_new_tokens) is not int or max_new_tokens < 1 or type(lookahead) is not int or lookahead < 1:
        raise ValueError("nonempty prompts and positive budgets required")
    if target.training:
        raise ValueError("target must be in eval mode")
    params = params or SamplingParams()
    result = BatchSpecResult([list(p) for p in prompts], [[] for _ in prompts], [[] for _ in prompts])
    caches = []
    for prompt in prompts:
        capacity = len(prompt) + max_new_tokens
        if capacity > target.config.max_seq_len:
            raise ValueError("generation budget exceeds target context")
        caches.append(PagedCache(target_pool) if target_pool is not None else target.make_kv_caches(1, capacity))
    rngs = [torch.Generator(device=target.device).manual_seed(seed + i) for i in range(len(prompts))]
    next_logits, pending = {}, {}
    active = list(range(len(prompts)))

    def target_call(indices, rows, last_only=False):
        chunks = [torch.tensor([row], device=target.device) for row in rows]
        selected_caches = [caches[i] for i in indices]
        if hasattr(target, "forward_batch"):
            outputs = target.forward_batch(chunks, selected_caches, last_only=last_only)
        else:
            outputs = [target(chunk, caches=cache) for chunk, cache in zip(chunks, selected_caches)]
            if last_only:
                outputs = [output[:, -1:] for output in outputs]
        result.target_calls += 1
        return {i: output[0] for i, output in zip(indices, outputs)}

    def emit(index, token):
        result.tokens[index].append(token)
        result.generated[index].append(token)
        result.token_times[index].append(time.perf_counter())
        return token == eos_token_id or len(result.generated[index]) == max_new_tokens

    try:
        prefill = target_call(active, prompts, last_only=True)
        proposer.prepare(prompts, max_new_tokens, params, rngs)
        if target.device.type == "cuda":
            torch.cuda.synchronize(target.device)
        result.prefill_done_at = time.perf_counter()
        for index in active:
            next_logits[index] = prefill[index][-1]
            pending[index] = None
        while active:
            base = {i: len(result.tokens[i]) for i in active}
            budgets = {i: min(lookahead, max_new_tokens - len(result.generated[i])) for i in active}
            proposals = proposer.propose(active, result.tokens, budgets, eos_token_id)
            verify_indices = [i for i in active if pending[i] is not None or proposals[i].tokens]
            verification = {}
            if verify_indices:
                rows = [(([pending[i]] if pending[i] is not None else []) + proposals[i].tokens)
                        for i in verify_indices]
                verification = target_call(verify_indices, rows)

            continuing, commits, accepted_round = [], [], {}
            for index in active:
                proposal = proposals[index]
                verified = verification.get(index)
                if pending[index] is not None:
                    next_logits[index], verified = verified[0], verified[1:]
                result.proposed += len(proposal.tokens)
                accepted = 0
                replacement = None
                stopped = False
                for offset, token in enumerate(proposal.tokens):
                    logits = next_logits[index] if offset == 0 else verified[offset - 1]
                    accepted_token, correction = _accept_or_correct(
                        logits, token, proposal.distributions[offset], params, rngs[index])
                    if not accepted_token:
                        replacement = correction
                        break
                    accepted += 1
                    if emit(index, token):
                        stopped = True
                        break
                result.accepted += accepted
                accepted_round[str(index)] = accepted
                target.rollback(caches[index], base[index] + accepted)
                if not stopped and replacement is None:
                    bonus_logits = next_logits[index] if not proposal.tokens else verified[-1]
                    replacement = (int(bonus_logits.argmax().item()) if params.temperature == 0
                                   else draw(params.probabilities(bonus_logits), rngs[index]))
                    if proposal.tokens:
                        result.bonus_tokens += 1
                if not stopped:
                    stopped = emit(index, replacement)
                commits.append(Commit(index, base[index], accepted, None if stopped else replacement, stopped))
                if not stopped:
                    pending[index] = replacement
                    continuing.append(index)
            proposer.commit(commits)
            result.rounds.append(accepted_round)
            active = continuing
        result.proposer_calls = proposer.calls
        result.draft_calls = proposer.calls if isinstance(proposer, DraftModelProposer) else 0
        return result
    finally:
        proposer.close()
        for cache in caches:
            if isinstance(cache, PagedCache):
                cache.close()


def speculative_batch(target, draft, prompts, max_new_tokens=32, lookahead=4, params=None, seed=0,
                      target_pool=None, draft_pool=None, eos_token_id=None):
    proposer = DraftModelProposer(target, draft, draft_pool)
    return speculative_batch_with_proposer(target, proposer, prompts, max_new_tokens, lookahead,
                                           params, seed, target_pool, eos_token_id)
