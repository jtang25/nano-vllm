import tempfile
import unittest
from pathlib import Path

import torch

from scripts.llama_export import export_llama
from test_inference import tiny


class LlamaExportTests(unittest.TestCase):
    @torch.inference_mode()
    def test_logits_match_transformers_llama(self):
        try:
            from transformers import LlamaForCausalLM
            import safetensors  # noqa: F401
        except ImportError:
            self.skipTest("install transformers and safetensors for independent model parity")
        model = tiny()
        ids = torch.tensor([[1, 4, 2, 3, 6, 5]])
        with tempfile.TemporaryDirectory() as directory:
            export_llama(model, directory)
            reference = LlamaForCausalLM.from_pretrained(Path(directory), attn_implementation="eager").eval()
            torch.testing.assert_close(model(ids), reference(ids).logits, atol=2e-5, rtol=1e-4)
