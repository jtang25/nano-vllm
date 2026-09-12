import tempfile
import unittest
from pathlib import Path
import torch

from model.checkpoint import load_checkpoint, save_checkpoint
from model.config import ModelConfig
from model.layers import DecoderLM
from model.tokenizer import ByteTokenizer
from runtime.sampling import SamplingParams


class IOTests(unittest.TestCase):
    def test_tokenizer_unicode_roundtrip(self):
        tokenizer = ByteTokenizer()
        text = "Hello 世界 👋\n"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text, eos=True)), text)

    @torch.inference_mode()
    def test_checkpoint_roundtrip(self):
        model = DecoderLM(config=ModelConfig(d_model=32, n_layers=2, d_ff=48)).eval()
        ids = torch.tensor([[256, 65, 66]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(model, path, {"test": True})
            restored = load_checkpoint(path)
            torch.testing.assert_close(model(ids), restored(ids), atol=0, rtol=0)

    def test_sampling_support(self):
        logits = torch.tensor([0., 1., 2., 3.])
        p = SamplingParams(top_k=2).probabilities(logits)
        self.assertEqual(p[:2].sum(), 0)
        p = SamplingParams(top_p=.1).probabilities(logits)
        self.assertEqual(p.argmax(), 3)
        self.assertEqual(torch.count_nonzero(p), 1)
        self.assertEqual(SamplingParams(0).probabilities(logits)[3], 1)
