"""Measure the explainable PyTorch page-loop to bounded-page-tile optimization."""
import argparse
import json
import os
import statistics
import time
import torch

from model.checkpoint import load_checkpoint
from benchmarks.experiment_utils import environment, save
from benchmarks.profile_workload import sync
from storage.paged_cache import PagedCache
from scripts.run import make_pool


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()
    model = load_checkpoint(args.checkpoint, "cuda", torch.float16)
    ids = torch.ones((1, args.context), dtype=torch.long, device="cuda")
    token = torch.ones((1, 1), dtype=torch.long, device="cuda")
    runs = []
    for tile in (args.block_size, 1024):
        os.environ["NANOVLLM_ATTENTION_TILE_TOKENS"] = str(tile)
        samples = []
        for repeat in range(args.repeats + 1):
            pool = make_pool(model, (args.context + args.tokens + args.block_size - 1) // args.block_size, args.block_size)
            cache = PagedCache(pool)
            model(ids, caches=cache, last_only=True)
            sync(model.device)
            start = time.perf_counter()
            for _ in range(args.tokens):
                model(token, caches=cache, last_only=True)
            sync(model.device)
            if repeat:
                samples.append(time.perf_counter() - start)
            cache.close()
        runs.append({"tile_tokens": tile, "seconds": samples, "median_seconds": statistics.median(samples),
                     "median_tpot_ms": statistics.median(samples) * 1000 / args.tokens})
    report = {"environment": environment(model, args.checkpoint), "context": args.context, "tokens": args.tokens,
              "block_size": args.block_size, "repeats": args.repeats, "runs": runs,
              "speedup": runs[0]["median_seconds"] / runs[1]["median_seconds"],
              "method": "index-select bounded groups of physical pages, then combine tile results with online softmax"}
    save(args.output, report)
    print(json.dumps({"speedup": report["speedup"], "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
