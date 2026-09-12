import unittest

import torch

from runtime.generation import generate
from runtime.prompt_lookup import NgramProposer, lookup_candidate, lookup_match, prompt_lookup_generate
from runtime.proposers import Proposal
from runtime.sampling import SamplingParams
from runtime.speculative_batch import speculative_batch_with_proposer
from benchmarks.sweep_prompt_lookup import request_latency
from test_inference import tiny


class RaggedProposalStub:
    """Sequence zero is empty while sequence one proposes in every round."""

    def __init__(self):
        self.calls = 0

    def prepare(self, prompts, max_new_tokens, params, rngs):
        self.calls = 0

    def propose(self, indices, histories, budgets, eos_token_id=None):
        self.calls += 1
        return {i: (Proposal([], []) if i == 0 else Proposal([0] * budgets[i], [None] * budgets[i]))
                for i in indices}

    def commit(self, commits):
        pass

    def close(self):
        pass


class PromptLookupTests(unittest.TestCase):
    def test_per_request_latency_uses_corresponding_start(self):
        measured = request_latency([[11.0, 11.2], [21.0, 21.4]], [10.0, 20.0])
        self.assertEqual(measured["ttft_ms"], [1000.0, 1000.0])
        self.assertAlmostEqual(measured["median_tpot_ms"], 300.0)

    def test_longest_suffix_and_earliest_tie(self):
        match = lookup_match([1, 2, 9, 1, 2, 8, 1, 2], 3, 2, 3)
        self.assertEqual((match.start, match.ngram_size, match.tokens), (0, 2, [9, 1, 2]))
        self.assertEqual(lookup_candidate([7, 8, 9, 7, 8], 3, 2, 3), [9, 7, 8])
        self.assertEqual(lookup_candidate([1], 4, 1, 4), [])

    def test_shortens_to_minimum_and_searches_live_history(self):
        history = [4, 5, 6, 4, 7, 4]
        self.assertEqual(lookup_candidate(history, 2, 1, 3), [5, 6])
        self.assertEqual(lookup_candidate(history, 2, 2, 3), [])
        history.extend([5, 6, 4])
        self.assertEqual(lookup_candidate(history, 2, 3, 3), [7, 4])

    @torch.inference_mode()
    def test_verified_output_matches_target_greedy_and_emits_bonus(self):
        model = tiny()
        prompt = [1, 2, 3, 1, 2, 3, 1, 2]
        expected = generate(model, prompt, 20, SamplingParams(0)).tokens
        result = prompt_lookup_generate(model, prompt, 20, lookahead=4, min_ngram=1,
                                        max_ngram=4, params=SamplingParams(0))
        self.assertEqual(result.tokens, expected)
        self.assertGreater(result.proposed, 0)
        self.assertGreater(result.bonus_tokens, 0)

    @torch.inference_mode()
    def test_zero_proposal_mixed_with_nonempty_packed_verification(self):
        model = tiny()
        prompts = [[1, 2], [3, 4, 2, 1]]
        expected = [generate(model, prompt, 9, SamplingParams(0)).tokens for prompt in prompts]
        result = speculative_batch_with_proposer(model, RaggedProposalStub(), prompts, 9, 4,
                                                 SamplingParams(0))
        self.assertEqual(result.tokens, expected)
        self.assertGreater(result.proposed, 0)

    def test_ngram_proposer_declares_point_mass_distribution(self):
        proposer = NgramProposer(1, 4)
        proposer.prepare([[1, 2, 1]], 4, SamplingParams(), [torch.Generator()])
        proposal = proposer.propose([0], [[1, 2, 1]], {0: 2})[0]
        self.assertEqual(proposal.tokens, [2, 1])
        self.assertEqual(proposal.distributions, [None, None])

    @torch.inference_mode()
    def test_sampling_path_smoke(self):
        model = tiny()
        result = prompt_lookup_generate(model, [1, 2, 1, 2], 7, lookahead=3,
                                        params=SamplingParams(.8, top_k=8, top_p=.9), seed=12)
        self.assertEqual(len(result.generated), 7)


if __name__ == "__main__":
    unittest.main()
