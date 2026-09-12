import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from model.checkpoint import load_checkpoint


class TrainingTests(unittest.TestCase):
    def test_train_save_load_and_generate(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "text.txt"
            checkpoint = Path(directory) / "tiny.pt"
            corpus.write_text("Small models learn repeated text.\n" * 100, encoding="utf-8")
            command = [sys.executable, "-m", "scripts.train", "--text", str(corpus), "--output", str(checkpoint),
                       "--steps", "2", "--seq-len", "8", "--batch-size", "2", "--dim", "32",
                       "--layers", "1", "--experts", "3", "--threads", "1"]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            model = load_checkpoint(checkpoint)
            self.assertEqual(model.config.n_experts, 3)
            result = subprocess.run([sys.executable, "-m", "scripts.run", "--checkpoint", str(checkpoint),
                "--mode", "paged", "--tokens", "2", "--threads", "1"],
                cwd=root, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("forward_calls", result.stdout)
