import unittest
import torch

from runtime.engine import Engine
from storage.paged_cache import PagedCache
from runtime.sampling import SamplingParams
from test_inference import tiny, pool_for


class PrefixTests(unittest.TestCase):
    @torch.inference_mode()
    def test_partial_block_copy_on_write_and_parent_isolation(self):
        model = tiny()
        pool = pool_for(model, block_size=4)
        parent = PagedCache(pool)
        ids = torch.tensor([[1, 2, 3]])
        model(ids, caches=parent)
        child = parent.fork()
        original = parent.block_table[0]
        self.assertEqual(len(pool.owners[original]), 2)
        output = model(torch.tensor([[4, 5]]), caches=child)
        self.assertNotEqual(child.block_table[0], original)
        self.assertEqual(parent.block_table, [original])
        torch.testing.assert_close(output, model(torch.tensor([[1, 2, 3, 4, 5]]))[:, 3:], atol=2e-5, rtol=1e-4)
        output = model(torch.tensor([[6]]), caches=parent)
        torch.testing.assert_close(output, model(torch.tensor([[1, 2, 3, 6]]))[:, -1:], atol=2e-5, rtol=1e-4)
        parent.close()
        child.close()
        self.assertEqual(len(pool.free), pool.n_blocks)

    @torch.inference_mode()
    def test_cow_exhaustion_is_atomic(self):
        model = tiny()
        pool = pool_for(model, blocks=1, block_size=4)
        parent = PagedCache(pool)
        model(torch.tensor([[1, 2]]), caches=parent)
        child = parent.fork()
        with self.assertRaises(MemoryError):
            model(torch.tensor([[3]]), caches=child)
        self.assertEqual(child.length, 2)
        self.assertEqual(child.block_table, parent.block_table)
        self.assertEqual(len(pool.owners[0]), 2)
        child.close()
        parent.close()

    def test_engine_reuses_prefix(self):
        model = tiny()
        pool = pool_for(model)
        engine = Engine(model, pool, prefix_entries=4)
        first = engine.submit([1, 2, 3, 4], 2, SamplingParams(0))
        engine.run()
        second = engine.submit([1, 2, 3, 4], 2, SamplingParams(0))
        engine.run()
        self.assertEqual(engine.requests[first].tokens, engine.requests[second].tokens)
        self.assertEqual(engine.prefixes.hits, 1)
        self.assertEqual(engine.prefixes.reused_tokens, 3)
        engine.close()
        self.assertEqual(len(pool.free), pool.n_blocks)
