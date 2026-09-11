"""A fixed prefill/decode trace with NVTX ranges for Nsight Systems/Compute."""
import argparse
import json
import time
from pathlib import Path
from contextlib import contextmanager
import torch

from model.checkpoint import load_checkpoint
from benchmarks.experiment_utils import environment, save, sync
from storage.paged_cache import PagedCache
from scripts.run import make_pool
import math


@contextmanager
def region(name, device):
    with torch.profiler.record_function(name):
        if device.type == "cuda":
            torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            if device.type == "cuda":
                torch.cuda.nvtx.range_pop()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--torch-trace")
    parser.add_argument("--layout", choices=("paged", "contiguous"), default="paged")
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()
    model = load_checkpoint(args.checkpoint, args.device, getattr(torch, args.dtype))
    for layer in model.layers:
        layer.attention.backend = "sdpa"
    ids = torch.ones(args.batch, args.context, dtype=torch.long, device=model.device)
    token = torch.ones(args.batch, 1, dtype=torch.long, device=model.device)

    def execute(markers=False):
        if args.layout == "paged":
            pool = make_pool(model, args.batch * math.ceil((args.context + args.tokens) / args.block_size), args.block_size)
            caches = [PagedCache(pool) for _ in range(args.batch)]

            def forward(tokens):
                return model.forward_batch(list(tokens.split(1, dim=0)), caches, last_only=True)
        else:
            caches = model.make_kv_caches(args.batch, args.context + args.tokens)

            def forward(tokens):
                return model(tokens, caches=caches, last_only=True)
        if markers:
            with region("prefill", model.device):
                forward(ids)
        else:
            forward(ids)
        sync(model.device)
        start = time.perf_counter()
        if model.device.type == "cuda":
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        with region("decode" if markers else "warmup", model.device):
            for _ in range(args.tokens):
                forward(token)
        if model.device.type == "cuda":
            end.record()
        sync(model.device)
        wall_seconds = time.perf_counter() - start
        if args.layout == "paged":
            for cache in caches:
                cache.close()
        return {"decode_wall_seconds": wall_seconds,
                "decode_device_span_seconds": begin.elapsed_time(end) / 1000 if model.device.type == "cuda" else None}

    for _ in range(args.warmup):
        execute()
    if args.torch_trace:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if model.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
            timings = execute(True)
        Path(args.torch_trace).parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(args.torch_trace)
    else:
        timings = execute(True)
    report = {"environment": environment(model, args.checkpoint), "batch": args.batch, "context": args.context,
              "tokens": args.tokens, "cache_layout": args.layout, "attention": "sdpa-prefill", **timings,
              "note": "Profiled timings include instrumentation overhead; use unprofiled timings for reported latency."}
    save(args.output, report)
    print(json.dumps(timings))


if __name__ == "__main__":
    main()
