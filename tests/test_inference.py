import copy
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from model.config import ModelConfig
from model.layers import DecoderLM, MoE
from storage.paged_cache import BlockPool, PagedCache
from runtime.engine import Engine
from runtime.generation import generate
from runtime.sampling import SamplingParams, residual_distribution
from runtime.speculative import speculative_generate


def tiny(**kwargs):
    options = dict(vocab_size=19, d_model=32, n_layers=2, n_q_heads=4, n_kv_heads=2,
                   d_ff=48, max_seq_len=64)
    options.update(kwargs)
    return DecoderLM(config=ModelConfig(**options)).eval()


def pool_for(model, blocks=24, block_size=3):
    c = model.config
    return BlockPool(c.n_layers, blocks, block_size, c.n_kv_heads, c.head_dim,
                     device=model.device, dtype=model.dtype)


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(11)
        self.model = tiny()
        self.ids = torch.randint(0, 19, (1, 11))

    @torch.inference_mode()
    def test_all_cache_paths_and_head_counts(self):
        for hkv in (1, 2, 4):
            for experts in (0, 3):
                model = tiny(n_kv_heads=hkv, n_experts=experts)
                expected = model(self.ids)
                for chunks in ([11], [1] * 11, [2, 4, 1, 4]):
                    for paged in (False, True):
                        with self.subTest(hkv=hkv, experts=experts, chunks=chunks, paged=paged):
                            pool = pool_for(model)
                            cache = PagedCache(pool) if paged else model.make_kv_caches(1, 11)
                            pieces, start = [], 0
                            for size in chunks:
                                pieces.append(model(self.ids[:, start:start + size], caches=cache))
                                start += size
                            torch.testing.assert_close(torch.cat(pieces, 1), expected, atol=2e-5, rtol=1e-4)
                            if paged:
                                self.assertEqual(cache.length, 11)
                                self.assertEqual(cache.live_bytes, 2 * 2 * 11 * hkv * 8 * 4)
                                cache.close()
                                self.assertEqual(len(pool.free), pool.n_blocks)

    @torch.inference_mode()
    def test_sdpa_reference(self):
        reference = self.model(self.ids)
        for layer in self.model.layers:
            layer.attention.backend = "sdpa"
        torch.testing.assert_close(self.model(self.ids), reference, atol=2e-5, rtol=1e-4)
        cache = self.model.make_kv_caches(1, 11)
        self.model(self.ids[:, :6], caches=cache)
        torch.testing.assert_close(self.model(self.ids[:, 6:], caches=cache), reference[:, 6:], atol=2e-5, rtol=1e-4)

    @torch.inference_mode()
    def test_pool_noncontiguous_reuse_and_poison(self):
        pool = pool_for(self.model, blocks=8, block_size=3)
        a, b = PagedCache(pool), PagedCache(pool)
        self.model(self.ids[:, :3], caches=a)
        self.model(self.ids[:, :3], caches=b)
        self.model(self.ids[:, 3:6], caches=a)
        self.assertEqual(a.block_table, [0, 2])
        a.close()
        b.close()
        pool.k.fill_(float("nan"))
        pool.v.fill_(float("nan"))
        cache = PagedCache(pool)
        actual = self.model(self.ids[:, :4], caches=cache)
        torch.testing.assert_close(actual, self.model(self.ids[:, :4]), atol=2e-5, rtol=1e-4)
        with patch.object(cache.layers[0], "gather", side_effect=AssertionError("must not gather")):
            self.model(self.ids[:, 4:5], caches=cache)
        cache.close()
        cache.close()
        self.assertEqual(len(pool.free), 8)

    @torch.inference_mode()
    def test_failure_rollback_and_exhaustion(self):
        pool = pool_for(self.model, blocks=4)
        cache = PagedCache(pool)
        self.model(self.ids[:, :2], caches=cache)
        with patch.object(self.model.layers[1], "forward", side_effect=RuntimeError("failure")):
            with self.assertRaises(RuntimeError):
                self.model(self.ids[:, 2:7], caches=cache)
        self.assertEqual(cache.length, 2)
        self.assertEqual(len(cache.block_table), 1)
        actual = self.model(self.ids[:, 2:7], caches=cache)
        torch.testing.assert_close(actual, self.model(self.ids[:, :7])[:, 2:], atol=2e-5, rtol=1e-4)
        before = list(cache.block_table)
        with self.assertRaises(MemoryError):
            cache.reserve(30)
        self.assertEqual(cache.block_table, before)
        cache.truncate(1)
        self.assertEqual(len(cache.block_table), 1)
        cache.close()

    @torch.inference_mode()
    def test_packed_mixed_prefill_decode(self):
        pool = pool_for(self.model)
        a, b = PagedCache(pool), PagedCache(pool)
        self.model(self.ids[:, :5], caches=a)
        outputs = self.model.forward_batch([self.ids[:, 5:6], self.ids[:, :4]], [a, b])
        torch.testing.assert_close(outputs[0], self.model(self.ids[:, :6])[:, -1:], atol=2e-5, rtol=1e-4)
        torch.testing.assert_close(outputs[1], self.model(self.ids[:, :4]), atol=2e-5, rtol=1e-4)
        a.close()
        b.close()

    @torch.inference_mode()
    def test_engine_matches_independent_generation(self):
        pool = pool_for(self.model, blocks=16)
        engine = Engine(self.model, pool, max_batch_tokens=4, max_requests=3)
        expected = {}
        for i, prompt in enumerate(([1, 2, 3], [4], [7, 8, 9, 3, 2, 1])):
            request = engine.submit(prompt, 5, SamplingParams(0), seed=i)
            expected[request] = generate(self.model, prompt, 5, SamplingParams(0)).tokens
        engine.step()
        late = engine.submit([2, 4], 3, SamplingParams(0))
        expected[late] = generate(self.model, [2, 4], 3, SamplingParams(0)).tokens
        engine.run()
        for request_id, tokens in expected.items():
            self.assertEqual(engine.requests[request_id].tokens, tokens)
        self.assertEqual(len(pool.free), pool.n_blocks)
        self.assertEqual(engine.credits, 0)

    def test_cancel_and_waiting_capacity(self):
        pool = pool_for(self.model, blocks=4, block_size=4)
        engine = Engine(self.model, pool, max_batch_tokens=2)
        first = engine.submit([1, 2, 3, 4], 10)
        second = engine.submit([1], 5)
        engine.step()
        self.assertEqual(engine.requests[second].status, "waiting")
        engine.cancel(first)
        engine.run()
        self.assertEqual(engine.requests[second].status, "finished")
        self.assertEqual(len(pool.free), 4)

    def test_engine_failure_after_another_request_finishes(self):
        pool = pool_for(self.model)
        engine = Engine(self.model, pool)
        first = engine.submit([1], 1, SamplingParams(0))
        second = engine.submit([2], 2, SamplingParams(0))
        with patch("runtime.engine.draw", side_effect=[3, RuntimeError("injected sampling failure")]):
            with self.assertRaises(RuntimeError):
                engine.step()
        self.assertEqual(engine.requests[first].status, "finished")
        self.assertEqual(engine.requests[second].status, "failed")
        self.assertEqual(engine.credits, 0)
        self.assertEqual(len(pool.free), pool.n_blocks)

    @torch.inference_mode()
    def test_speculative_greedy_and_identical_draft(self):
        expected = generate(self.model, [1, 2, 3], 12, SamplingParams(0)).tokens
        for draft in (copy.deepcopy(self.model), tiny(n_layers=1)):
            for lookahead in (1, 3, 8):
                for paged in (False, True):
                    tp, dp = (pool_for(self.model), pool_for(draft)) if paged else (None, None)
                    result = speculative_generate(self.model, draft, [1, 2, 3], 12, lookahead,
                        SamplingParams(0), target_pool=tp, draft_pool=dp)
                    self.assertEqual(result.tokens, expected)
                    self.assertEqual(len(result.generated), 12)
                    if tp:
                        self.assertEqual(len(tp.free), tp.n_blocks)
                        self.assertEqual(len(dp.free), dp.n_blocks)

    @torch.inference_mode()
    def test_speculative_eos_and_zero(self):
        eos = generate(self.model, [1, 2], 1, SamplingParams(0)).generated[0]
        result = speculative_generate(self.model, tiny(n_layers=1), [1, 2], 8, 4, SamplingParams(0), eos)
        self.assertEqual(result.generated, [eos])
        result = speculative_generate(self.model, self.model, [1, 2], 0)
        self.assertEqual(result.target_calls, 0)

    def test_rejection_distribution_identity(self):
        p = torch.tensor([0.1, 0.6, 0.3], dtype=torch.float64)
        q = torch.tensor([0.5, 0.2, 0.3], dtype=torch.float64)
        accepted_mass = torch.minimum(p, q)
        rejection_mass = 1 - accepted_mass.sum()
        actual = accepted_mass + rejection_mass * residual_distribution(p, q)
        torch.testing.assert_close(actual, p, atol=1e-14, rtol=0)
        # Covers disjoint support too.
        p, q = torch.tensor([1., 0.]), torch.tensor([0., 1.])
        torch.testing.assert_close(residual_distribution(p, q), p)

    def test_moe_dispatch_against_dense_expert_evaluation(self):
        moe = MoE(8, 12, 3, 2)
        x = torch.randn(2, 4, 8)
        logits = moe.router(x)
        values, indices = logits.topk(2, dim=-1)
        weights = values.softmax(-1)
        expected = torch.zeros_like(x)
        for i, expert in enumerate(moe.experts):
            routing = ((indices == i) * weights).sum(-1, keepdim=True)
            expected = expected + routing * expert(x)
        torch.testing.assert_close(moe(x), expected)
        (moe(x).square().mean() + .01 * moe.balance_loss(x)).backward()
        self.assertIsNotNone(moe.router.weight.grad)

    def test_training_backward(self):
        model = tiny(n_experts=3).train()
        logits = model(self.ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 19), self.ids[:, 1:].reshape(-1))
        loss.backward()
        self.assertTrue(torch.isfinite(model.token_embedding.weight.grad).all())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA correctness requires a GPU; no timing tests")
class CudaTests(unittest.TestCase):
    @torch.inference_mode()
    def test_cuda_shared_prefix_and_batched_speculation(self):
        from runtime.speculative_batch import speculative_batch
        torch.manual_seed(42)
        model = tiny().to(device="cuda", dtype=torch.float16)
        draft = tiny(n_layers=1).to(device="cuda", dtype=torch.float16)
        pool = pool_for(model, blocks=32, block_size=4)
        parent = PagedCache(pool)
        model(torch.tensor([[1, 2, 3]], device="cuda"), caches=parent)
        child = parent.fork()
        result = model(torch.tensor([[4, 5]], device="cuda"), caches=child)
        expected = model(torch.tensor([[1, 2, 3, 4, 5]], device="cuda"))[:, 3:]
        torch.testing.assert_close(result, expected, atol=.01, rtol=.01)
        self.assertNotEqual(child.block_table[0], parent.block_table[0])
        parent.close()
        child.close()
        prompts = [[1, 2], [3, 4, 5]]
        expected = [generate(model, p, 8, SamplingParams(0)).tokens for p in prompts]
        result = speculative_batch(model, draft, prompts, 8, 4, SamplingParams(0),
                                   target_pool=pool, draft_pool=pool_for(draft, blocks=32))
        self.assertEqual(result.tokens, expected)
        self.assertEqual(len(pool.free), pool.n_blocks)

    @torch.inference_mode()
    def test_cuda_engine_and_speculative(self):
        model = tiny().to(device="cuda", dtype=torch.float16)
        draft = tiny(n_layers=1).to(device="cuda", dtype=torch.float16)
        expected = generate(model, [1, 2], 5, SamplingParams(0)).tokens
        pool = pool_for(model)
        engine = Engine(model, pool, max_batch_tokens=3)
        request = engine.submit([1, 2], 5, SamplingParams(0))
        engine.submit([3, 4, 5], 4, SamplingParams(0))
        engine.run()
        self.assertEqual(engine.requests[request].tokens, expected)
        result = speculative_generate(model, draft, [1, 2], 5, params=SamplingParams(0),
                                      target_pool=pool, draft_pool=pool_for(draft))
        self.assertEqual(result.tokens, expected)

    @torch.inference_mode()
    def test_cuda_dtypes_paged_sdpa_moe(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            model = tiny(n_experts=3).to(device="cuda", dtype=dtype)
            ids = torch.tensor([[1, 3, 2, 4, 5]], device="cuda")
            reference = model(ids)
            pool = pool_for(model)
            cache = PagedCache(pool)
            parts = [model(ids[:, :3], caches=cache), model(ids[:, 3:], caches=cache)]
            torch.testing.assert_close(torch.cat(parts, 1), reference, atol=.03, rtol=.03)
            for layer in model.layers:
                layer.attention.backend = "sdpa"
            torch.testing.assert_close(model(ids), reference, atol=.03, rtol=.03)
            cache.close()


if __name__ == "__main__":
    unittest.main()
