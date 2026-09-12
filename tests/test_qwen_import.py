import tempfile
import unittest
from pathlib import Path
import torch

from model.checkpoint import save_checkpoint, load_checkpoint
from scripts.qwen_import import native_config, from_state
from storage.paged_cache import PagedCache
from scripts.run import make_pool
from model.tokenizer import check_tokenizers


class QwenImportTests(unittest.TestCase):
    @torch.inference_mode()
    def test_import_reference_cache_and_checkpoint(self):
        try:
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ImportError:
            self.skipTest("transformers required for independent Qwen parity")
        torch.manual_seed(42)
        c = Qwen2Config(vocab_size=41, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64, tie_word_embeddings=True)
        c._attn_implementation = "eager"
        reference = Qwen2ForCausalLM(c).eval()
        model = from_state(reference.state_dict(), native_config(c.to_dict()))
        ids = torch.tensor([[1, 5, 10, 3, 2, 9]])
        expected = reference(ids).logits
        torch.testing.assert_close(model(ids), expected, atol=2e-6, rtol=2e-5)
        cache = PagedCache(make_pool(model, 8, 2))
        output = torch.cat([model(ids[:, :3], caches=cache), model(ids[:, 3:5], caches=cache), model(ids[:, 5:], caches=cache)], 1)
        torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-5)
        cache.close()
        model.tokenizer_spec = {"type": "huggingface", "repo_id": "test", "revision": "test", "backend_sha256": "test"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(model, path)
            restored = load_checkpoint(path)
            self.assertIs(restored.lm_head.weight, restored.token_embedding.weight)
            self.assertEqual(restored.tokenizer_spec, model.tokenizer_spec)
            torch.testing.assert_close(restored(ids), expected, atol=2e-6, rtol=2e-5)
            restored.tokenizer_spec = {**model.tokenizer_spec, "backend_sha256": "different"}
            with self.assertRaises(ValueError):
                check_tokenizers(model, restored)

    def test_trim_padded_vocabulary(self):
        try:
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ImportError:
            self.skipTest("transformers required for independent Qwen parity")
        torch.manual_seed(7)
        c = Qwen2Config(vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32)
        reference = Qwen2ForCausalLM(c).eval()
        model = from_state(reference.state_dict(), native_config(c.to_dict(), vocab_size=41))
        ids = torch.tensor([[1, 7, 20, 40]])
        torch.testing.assert_close(model(ids), reference(ids).logits[..., :41], atol=2e-6, rtol=2e-5)

    def test_reject_unsupported_attention(self):
        with self.assertRaises(ValueError):
            native_config({"model_type": "qwen2", "use_sliding_window": True})
