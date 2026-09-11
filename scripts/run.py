"""Generate text with an ordinary cache, paged engine, or speculative decoder."""
import argparse
import json
import torch

from model.checkpoint import load_checkpoint
from model.config import ModelConfig
from runtime.engine import Engine
from runtime.generation import generate
from model.layers import DecoderLM
from storage.paged_cache import BlockPool
from runtime.prompt_lookup import prompt_lookup_generate
from runtime.sampling import SamplingParams
from runtime.speculative import speculative_generate
from model.tokenizer import tokenizer_for


def make_pool(model, blocks, block_size):
    c = model.config
    return BlockPool(c.n_layers, blocks, block_size, c.n_kv_heads, c.head_dim, device=model.device, dtype=model.dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--draft-checkpoint")
    parser.add_argument("--prompt", action="append", default=None, help="repeat for engine batching")
    parser.add_argument("--mode", choices=("dense", "paged", "engine", "speculative", "prompt-lookup"), default="paged")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=.8)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float, default=1.)
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--min-ngram", type=int, default=1)
    parser.add_argument("--max-ngram", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--batch-tokens", type=int, default=128)
    parser.add_argument("--prefix-entries", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    model = load_checkpoint(args.checkpoint, args.device, dtype) if args.checkpoint else DecoderLM().to(device=args.device, dtype=dtype).eval()
    if not args.checkpoint:
        print("Random weights: this is an execution demo, not a language-quality demo.")
    tokenizer = tokenizer_for(model)
    if model.vocab_size < tokenizer.vocab_size:
        parser.error("tokenizer exceeds model vocabulary")
    prompts = [tokenizer.encode(p) for p in (args.prompt or ["Hello"])]
    params = SamplingParams(args.temperature, args.top_k, args.top_p)
    pool = make_pool(model, args.blocks, args.block_size)
    if args.mode == "engine":
        engine = Engine(model, pool, args.batch_tokens, prefix_entries=args.prefix_entries, decode_first=True)
        try:
            for i, prompt in enumerate(prompts):
                engine.submit(prompt, args.tokens, params, tokenizer.eos_id, args.seed + i)
            for request in engine.run().values():
                print(json.dumps({"id": request.id, "text": tokenizer.decode(request.tokens), "tokens": request.generated}))
        finally:
            engine.close()
    elif args.mode == "speculative":
        draft = load_checkpoint(args.draft_checkpoint, args.device, dtype) if args.draft_checkpoint else DecoderLM(config=ModelConfig(n_layers=1)).to(device=args.device, dtype=dtype).eval()
        for prompt in prompts:
            result = speculative_generate(model, draft, prompt, args.tokens, args.lookahead, params,
                tokenizer.eos_id, args.seed, pool, make_pool(draft, args.blocks, args.block_size))
            print(json.dumps({"text": tokenizer.decode(result.tokens), **result.__dict__}))
    elif args.mode == "prompt-lookup":
        for prompt in prompts:
            result = prompt_lookup_generate(model, prompt, args.tokens, args.lookahead, args.max_ngram,
                args.min_ngram, params, tokenizer.eos_id, args.seed, pool)
            print(json.dumps({"text": tokenizer.decode(result.tokens), **result.__dict__}))
    else:
        for prompt in prompts:
            result = generate(model, prompt, args.tokens, params, tokenizer.eos_id, args.seed,
                              pool if args.mode == "paged" else None)
            print(json.dumps({"text": tokenizer.decode(result.tokens), **result.__dict__}))


if __name__ == "__main__":
    main()
