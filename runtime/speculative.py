"""Single-request adapters over the shared batched speculative verifier."""
from dataclasses import dataclass, field

from runtime.proposers import DraftModelProposer
from runtime.speculative_batch import speculative_batch_with_proposer


@dataclass
class SpeculativeResult:
    tokens: list[int]
    generated: list[int]
    token_times: list[float] = field(default_factory=list)
    target_calls: int = 0
    draft_calls: int = 0
    proposer_calls: int = 0
    proposed: int = 0
    accepted: int = 0
    bonus_tokens: int = 0
    accepted_per_round: list[int] = field(default_factory=list)


def speculative_generate_with_proposer(target, proposer, prompt, max_new_tokens=32, lookahead=4,
                                       params=None, eos_token_id=None, seed=0, target_pool=None):
    if max_new_tokens == 0:
        return SpeculativeResult(list(prompt), [])
    batch = speculative_batch_with_proposer(target, proposer, [prompt], max_new_tokens, lookahead,
                                            params, seed, target_pool, eos_token_id)
    return SpeculativeResult(batch.tokens[0], batch.generated[0], batch.token_times[0], batch.target_calls,
                             batch.draft_calls, batch.proposer_calls, batch.proposed, batch.accepted,
                             batch.bonus_tokens, [row["0"] for row in batch.rounds])


def speculative_generate(target, draft, prompt, max_new_tokens=32, lookahead=4, params=None,
                         eos_token_id=None, seed=0, target_pool=None, draft_pool=None):
    proposer = DraftModelProposer(target, draft, draft_pool)
    return speculative_generate_with_proposer(target, proposer, prompt, max_new_tokens, lookahead,
                                              params, eos_token_id, seed, target_pool)


def prompt_lookup_generate(target, prompt, max_new_tokens=32, lookahead=4, min_ngram=1, max_ngram=4,
                           params=None, eos_token_id=None, seed=0, target_pool=None):
    from runtime.prompt_lookup import NgramProposer
    proposer = NgramProposer(min_ngram, max_ngram)
    return speculative_generate_with_proposer(target, proposer, prompt, max_new_tokens, lookahead,
                                              params, eos_token_id, seed, target_pool)
