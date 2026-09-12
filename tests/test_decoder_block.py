import unittest

import torch

from model.attention import GQA
from storage.cache import KVCache
from model.layers import DecoderBlock


class DecoderBlockTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.d_model = 32
        self.block = DecoderBlock(self.d_model, n_q_heads=4, n_kv_heads=2, d_ff=64).eval()
        self.x = torch.randn(2, 6, self.d_model)

    @torch.inference_mode()
    def test_cached_attention_matches_uncached_reference(self):
        attention = self.block.attention
        positions = torch.arange(self.x.shape[1])
        reference = attention(self.x, positions)
        cache = KVCache(2, 2, 8, 8, device=self.x.device, dtype=self.x.dtype)
        outputs = []

        start = 0
        for size in (3, 1, 2):
            end = start + size
            outputs.append(attention(self.x[:, start:end], positions[start:end], cache))
            start = end

        torch.testing.assert_close(torch.cat(outputs, dim=1), reference, atol=1e-5, rtol=1e-4)
        self.assertEqual(cache.length, 6)

    @torch.inference_mode()
    def test_cached_block_matches_uncached_reference(self):
        positions = torch.arange(self.x.shape[1])
        reference = self.block(self.x, positions)
        cache = KVCache(2, 2, 8, 8, device=self.x.device, dtype=self.x.dtype)
        outputs = []

        for start, end in ((0, 2), (2, 3), (3, 6)):
            outputs.append(self.block(self.x[:, start:end], positions[start:end], cache))

        torch.testing.assert_close(torch.cat(outputs, dim=1), reference, atol=1e-5, rtol=1e-4)
        self.assertEqual(cache.length, 6)

    def test_cache_capacity(self):
        cache = KVCache(1, 2, 2, 8, device="cpu", dtype=torch.float32)
        k = torch.randn(1, 2, 2, 8)
        cache.append(k, k)

        with self.assertRaisesRegex(ValueError, "capacity"):
            cache.append(k[:, :, :1], k[:, :, :1])

    def test_rope_preserves_norm(self):
        attention = GQA(32, 4, 2)
        x = torch.randn(2, 4, 6, 8)
        positions = torch.arange(6)
        angles = positions.float()[:, None] * attention.inv_freq[None]
        rotated = attention._apply_rope(x, angles.cos()[None, None], angles.sin()[None, None])
        torch.testing.assert_close(rotated.square().sum(-1), x.square().sum(-1))


if __name__ == "__main__":
    unittest.main()
