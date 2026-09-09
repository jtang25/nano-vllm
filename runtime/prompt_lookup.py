"""N-gram proposals for prompt-lookup speculative decoding."""
from dataclasses import dataclass

from runtime.proposers import Proposal


@dataclass(frozen=True)
class LookupMatch:
    start: int
    ngram_size: int
    tokens: list[int]


def lookup_match(tokens, max_tokens=4, min_ngram=1, max_ngram=4):
    """Find the earliest previous occurrence of the longest matching suffix."""
    if type(max_tokens) is not int or max_tokens < 1:
        return None
    if type(min_ngram) is not int or type(max_ngram) is not int or min_ngram < 1 or max_ngram < min_ngram:
        raise ValueError("expected 1 <= min_ngram <= max_ngram")
    largest = min(max_ngram, len(tokens) - 1)
    for size in range(largest, min_ngram - 1, -1):
        suffix_start = len(tokens) - size
        suffix = tokens[suffix_start:]
        for start in range(suffix_start):
            if tokens[start:start + size] != suffix:
                continue
            candidate_start = start + size
            candidate = tokens[candidate_start:min(candidate_start + max_tokens, len(tokens))]
            if candidate:
                return LookupMatch(start, size, candidate)
    return None


def lookup_candidate(tokens, max_tokens=4, min_ngram=1, max_ngram=4):
    match = lookup_match(tokens, max_tokens, min_ngram, max_ngram)
    return [] if match is None else match.tokens


class NgramProposer:
    """Deterministic proposals copied from each request's full token history."""

    def __init__(self, min_ngram=1, max_ngram=4):
        if type(min_ngram) is not int or type(max_ngram) is not int or min_ngram < 1 or max_ngram < min_ngram:
            raise ValueError("expected 1 <= min_ngram <= max_ngram")
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        self.calls = 0
        self.matches = []

    def prepare(self, prompts, max_new_tokens, params, rngs):
        self.calls = 0
        self.matches = []

    def propose(self, indices, histories, budgets, eos_token_id=None):
        proposals = {}
        for index in indices:
            match = lookup_match(histories[index], budgets[index], self.min_ngram, self.max_ngram)
            tokens = [] if match is None else match.tokens
            if eos_token_id in tokens:
                tokens = tokens[:tokens.index(eos_token_id) + 1]
            proposals[index] = Proposal(tokens, [None] * len(tokens))
            if match is not None:
                self.matches.append({"sequence": index, "start": match.start,
                                     "ngram_size": match.ngram_size, "proposed": len(tokens)})
        self.calls += 1
        return proposals

    def commit(self, commits):
        pass

    def close(self):
        pass


def prompt_lookup_generate(target, prompt, max_new_tokens=32, lookahead=4, max_ngram=4,
                           min_ngram=1, params=None, eos_token_id=None, seed=0, pool=None):
    """Compatibility wrapper over the shared speculative verifier."""
    from runtime.speculative import prompt_lookup_generate as generate
    return generate(target, prompt, max_new_tokens, lookahead, min_ngram, max_ngram,
                    params, eos_token_id, seed, pool)
