"""Train a byte-level dense or MoE model from a local UTF-8 text corpus."""
import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from model.checkpoint import load_checkpoint, save_checkpoint
from model.config import ModelConfig
from model.layers import DecoderLM, MoE
from model.tokenizer import ByteTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/model.pt"))
    parser.add_argument("--init-checkpoint", help="initialize weights/geometry from a native checkpoint; optimizer starts fresh")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--experts", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.seq_len, args.threads) < 1:
        parser.error("training sizes must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    data = torch.tensor(ByteTokenizer().encode(args.text.read_text(encoding="utf-8"), eos=True), dtype=torch.long)
    split = int(len(data) * .9)
    training, validation = data[:split], data[split:]
    if min(len(training), len(validation)) <= args.seq_len:
        parser.error("corpus must provide more than seq-len bytes in both the 90% train and 10% validation splits")
    config = ModelConfig(d_model=args.dim, n_layers=args.layers, n_q_heads=args.heads,
                         n_kv_heads=args.kv_heads, d_ff=args.dim * 3, max_seq_len=max(2048, args.seq_len),
                         n_experts=args.experts, attention_backend="sdpa")
    model = load_checkpoint(args.init_checkpoint, args.device).train() if args.init_checkpoint else DecoderLM(config=config).to(args.device).train()
    config = model.config
    if isinstance(getattr(model, "tokenizer_spec", None), dict):
        parser.error("byte-corpus training is not compatible with an imported HF tokenizer")
    if args.seq_len > config.max_seq_len or config.vocab_size < ByteTokenizer.vocab_size:
        parser.error("checkpoint cannot accommodate the byte corpus or sequence length")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.1)
    rng = torch.Generator().manual_seed(args.seed)
    val_rng = torch.Generator().manual_seed(args.seed + 1)
    auxiliary = []

    def capture_balance(module, inputs, output):
        auxiliary.append(module.balance_loss(inputs[0]))

    handles = [module.register_forward_hook(capture_balance) for module in model.modules() if isinstance(module, MoE)]

    def batch(source, generator):
        starts = torch.randint(len(source) - args.seq_len, (args.batch_size,), generator=generator)
        x = torch.stack([source[s:s + args.seq_len] for s in starts])
        y = torch.stack([source[s + 1:s + args.seq_len + 1] for s in starts])
        return x.to(args.device), y.to(args.device)

    history = []
    for step in range(1, args.steps + 1):
        x, y = batch(training, rng)
        optimizer.zero_grad(set_to_none=True)
        auxiliary.clear()
        logits = model(x)
        ce = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
        loss = ce + .01 * torch.stack(auxiliary).mean() if auxiliary else ce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vx, vy = batch(validation, val_rng)
                val = F.cross_entropy(model(vx).reshape(-1, config.vocab_size), vy.reshape(-1))
            model.train()
            row = {"step": step, "train_cross_entropy": ce.item(), "validation_cross_entropy": val.item()}
            history.append(row)
            print(json.dumps(row), flush=True)
    for handle in handles:
        handle.remove()
    save_checkpoint(model, args.output, {"seed": args.seed, "steps": args.steps, "losses": history, "synthetic_weights": False})
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
