import unittest
import torch

from runtime.generation import generate
from runtime.sampling import SamplingParams
from runtime.speculative_batch import speculative_batch
from test_inference import tiny, pool_for


class BatchSpecTests(unittest.TestCase):
    @torch.inference_mode()
    def test_ragged_batch_matches_target(self):
        torch.manual_seed(4)
        target, draft = tiny(), tiny(n_layers=1)
        prompts = [[1, 2], [3, 4, 2, 1], [1]]
        expected = [generate(target, p, 9, SamplingParams(0)).tokens for p in prompts]
        for paged in (False, True):
            tp, dp = (pool_for(target, 32), pool_for(draft, 32)) if paged else (None, None)
            result = speculative_batch(target, draft, prompts, 9, 4, SamplingParams(0), target_pool=tp, draft_pool=dp)
            self.assertEqual(result.tokens, expected)
            self.assertEqual(result.target_calls, len(result.rounds) + 1)
            if paged:
                self.assertEqual(len(tp.free), tp.n_blocks)
                self.assertEqual(len(dp.free), dp.n_blocks)
