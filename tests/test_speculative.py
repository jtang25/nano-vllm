import types
import unittest
from unittest.mock import patch

import torch

from runtime.sampling import SamplingParams
from runtime.speculative import speculative_generate
from test_inference import tiny


class ToyCache:
    def __init__(self):
        self.length = 0

    def truncate(self, length):
        if not 0 <= length <= self.length:
            raise AssertionError("invalid rollback")
        self.length = length


class ToyModel:
    """Known distributions let us test sampling without neural-network noise."""
    def __init__(self, probabilities):
        self.device = torch.device("cpu")
        self.vocab_size = len(probabilities)
        self.training = False
        self.config = types.SimpleNamespace(max_seq_len=100)
        self.logits = torch.tensor(probabilities).log()

    def _validate_ids(self, ids):
        pass

    def make_kv_caches(self, batch, capacity):
        return [ToyCache()]

    def __call__(self, ids, caches):
        caches[0].length += ids.shape[1]
        return self.logits.expand(1, ids.shape[1], -1)

    def rollback(self, caches, length):
        caches[0].truncate(length)


class SpeculativeTests(unittest.TestCase):
    def test_sampled_distribution_including_second_token(self):
        target = ToyModel([.15, .55, .30])
        draft = ToyModel([.60, .10, .30])
        counts = torch.zeros(2, 3)
        for seed in range(1800):
            result = speculative_generate(target, draft, [0], 2, lookahead=2, seed=seed)
            for position, token in enumerate(result.generated):
                counts[position, token] += 1
        torch.testing.assert_close(counts / 1800, torch.tensor([[.15, .55, .30]]).expand(2, -1), atol=.04, rtol=0)

    @torch.inference_mode()
    def test_every_rejection_position_and_pending_alignment(self):
        # Target always chooses 1. Draft disagrees at one chosen absolute position.
        for reject_at in range(4):
            target, draft = tiny(), tiny(n_layers=1)
            target_forward, draft_forward = target.forward_batch, draft.forward_batch

            def scripted(original, disagreement=None):
                def forward(chunks, caches, last_only=False):
                    starts = [cache[0].length for cache in caches]
                    outputs = original(chunks, caches, last_only)
                    scripted_outputs = []
                    for chunk, start, output in zip(chunks, starts, outputs):
                        logits = torch.full_like(output, -10)
                        positions = [start + chunk.shape[1] - 1] if last_only else range(start, start + chunk.shape[1])
                        for column, position in enumerate(positions):
                            token = 2 if position == disagreement else 1
                            logits[:, column, token] = 10
                        scripted_outputs.append(logits)
                    return scripted_outputs
                return forward

            with patch.object(target, "forward_batch", side_effect=scripted(target_forward)):
                with patch.object(draft, "forward_batch", side_effect=scripted(draft_forward, 1 + reject_at)):
                    result = speculative_generate(target, draft, [3, 4], 10, 4, SamplingParams(0))
            self.assertEqual(result.generated, [1] * 10)
            self.assertEqual(result.accepted_per_round[0], reject_at)
            self.assertEqual(result.target_calls, 1 + len(result.accepted_per_round))

    def test_identical_distribution_accepts_all_proposals(self):
        target = ToyModel([.2, .3, .5])
        result = speculative_generate(target, target, [0], 10, 3, seed=17)
        self.assertEqual(result.accepted, result.proposed)

    def test_top_p_and_top_k_distribution(self):
        target = ToyModel([.1, .2, .3, .4])
        draft = ToyModel([.4, .3, .2, .1])
        params = SamplingParams(.8, top_k=2, top_p=.9)
        expected = params.probabilities(target.logits)
        counts = torch.zeros(4)
        for seed in range(1200):
            result = speculative_generate(target, draft, [0], 1, params=params, seed=seed)
            counts[result.generated[0]] += 1
        torch.testing.assert_close(counts / 1200, expected, atol=.05, rtol=0)
