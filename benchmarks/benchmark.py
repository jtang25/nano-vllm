"""Reproducible timing harness. CUDA runs are opt-in and never happen in CI."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from model.checkpoint import load_checkpoint
from model.config import ModelConfig
from runtime.engine import Engine
from runtime.generation import generate
from model.layers import DecoderLM
from storage.paged_cache import PagedCache
from scripts.run import make_pool
from runtime.sampling import SamplingParams
from runtime.speculative import speculative_generate


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(operation, device):
    synchronize(device)
    start = time.perf_counter()
    result = operation()
    synchronize(device)
    return result, (time.perf_counter() - start) * 1000


def summary(samples):
    ordered = sorted(samples)
    return {"median_ms": statistics.median(samples), "min_ms": min(samples),
            "p95_ms": ordered[min(len(ordered) - 1, int(.95 * len(ordered)))], "samples_ms": samples}


@torch.inference_mode()
def benchmark(model, prompt_length, decode_tokens, repeats, warmup, block_size, modes):
    if prompt_length + decode_tokens > model.config.max_seq_len:
        raise ValueError("benchmark exceeds model context")
    ids = torch.randint(model.vocab_size, (1, prompt_length + decode_tokens), device=model.device)
    expected = model(ids)
    results = {}
    for mode in modes:
        if mode not in ("uncached", "contiguous", "paged"):
            raise ValueError("invalid benchmark mode")
        prefill_times, decode_times, token_times = [], [], []
        pool = make_pool(model, (prompt_length + decode_tokens + block_size - 1) // block_size, block_size) if mode == "paged" else None
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model.device)
        for repeat in range(warmup + repeats):
            cache = None if mode == "uncached" else (PagedCache(pool) if pool is not None else model.make_kv_caches(1, prompt_length + decode_tokens))
            try:
                _, prefill_ms = timed(lambda: model(ids[:, :prompt_length], caches=None if mode == "uncached" else cache), model.device)
                elapsed = []
                for position in range(prompt_length, prompt_length + decode_tokens):
                    def step():
                        return model(ids[:, :position + 1])[:, -1:] if mode == "uncached" else model(ids[:, position:position + 1], caches=cache)
                    output, milliseconds = timed(step, model.device)
                    # Validation is outside the timing window; all modes use the same token stream.
                    torch.testing.assert_close(output, expected[:, position:position + 1], atol=.03 if model.dtype != torch.float32 else 2e-5, rtol=.03 if model.dtype != torch.float32 else 1e-4)
                    elapsed.append(milliseconds)
                if repeat >= warmup:
                    prefill_times.append(prefill_ms)
                    decode_times.append(sum(elapsed))
                    token_times.extend(elapsed)
            finally:
                if isinstance(cache, PagedCache):
                    cache.close()
                cache = None
        results[mode] = {"prefill": summary(prefill_times), "decode_total": summary(decode_times),
                         "time_per_decode_token": summary(token_times),
                         "decode_tokens_per_second": decode_tokens * 1000 / statistics.median(decode_times)}
        if model.device.type == "cuda":
            results[mode]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(model.device)
        del pool
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--checkpoint")
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--experts", type=int, default=0)
    parser.add_argument("--backend", choices=("reference", "sdpa"), default="sdpa")
    parser.add_argument("--modes", nargs="+", default=["uncached", "contiguous", "paged"])
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument("--profile", type=Path, help="optional Chrome trace output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--requests", type=int, default=4)
    args = parser.parse_args()
    if min(args.prompt_length, args.decode_tokens, args.repeats, args.block_size, args.requests) < 1 or args.warmup < 0:
        parser.error("sizes/repeats must be positive; warmup must be nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu --dtype float32 for a functional smoke run")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    model = load_checkpoint(args.checkpoint, device, dtype) if args.checkpoint else DecoderLM(config=ModelConfig(
        d_model=args.dim, n_layers=args.layers, d_ff=args.dim * 3, n_experts=args.experts,
        max_seq_len=max(2048, args.prompt_length + args.decode_tokens), attention_backend=args.backend)).to(device=device, dtype=dtype).eval()
    for layer in model.layers:
        layer.attention.backend = args.backend
    report = {"python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
              "dtype": str(dtype), "seed": args.seed, "threads": args.threads, "config": model.config.to_dict(),
              "attention_backend": args.backend, "checkpoint": args.checkpoint, "random_weights": args.checkpoint is None,
              "prompt_length": args.prompt_length, "decode_tokens": args.decode_tokens, "block_size": args.block_size,
              "repeats": args.repeats, "warmup": args.warmup}
    report["single_request"] = benchmark(model, args.prompt_length, args.decode_tokens, args.repeats, args.warmup, args.block_size, args.modes)
    c = model.config
    bytes_per_token = 2 * c.n_layers * c.n_kv_heads * c.head_dim * model.token_embedding.weight.element_size()
    report["memory_model"] = {"kv_bytes_per_token": bytes_per_token,
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "live_kv_bytes": bytes_per_token * (args.prompt_length + args.decode_tokens)}
    # A separately labelled end-to-end workload includes Python scheduling and sampling.
    blocks = args.requests * ((args.prompt_length + args.decode_tokens + args.block_size - 1) // args.block_size)
    pool = make_pool(model, blocks, args.block_size)
    prompts = [[1] * max(1, args.prompt_length - i) for i in range(args.requests)]

    def batched():
        engine = Engine(model, pool, max_batch_tokens=max(args.prompt_length, args.requests))
        try:
            for prompt in prompts:
                engine.submit(prompt, args.decode_tokens, SamplingParams(0))
            engine.run()
            latencies = []
            for request in engine.requests.values():
                latencies.append({
                    "ttft_ms": (request.first_token_at - request.submitted_at) * 1000,
                    "end_to_end_ms": (request.finished_at - request.submitted_at) * 1000,
                    "inter_token_ms": [(b - a) * 1000 for a, b in zip(request.token_times, request.token_times[1:])],
                })
            return engine.forward_calls, latencies
        finally:
            engine.close()

    engine_samples = []
    latency_samples = []
    for i in range(args.warmup + args.repeats):
        (calls, latencies), ms = timed(batched, device)
        if i >= args.warmup:
            engine_samples.append(ms)
            latency_samples.append(latencies)
    report["engine"] = {**summary(engine_samples), "requests": args.requests, "forward_calls": calls,
                        "request_latencies": latency_samples,
                        "output_tokens_per_second": args.requests * args.decode_tokens * 1000 / statistics.median(engine_samples)}
    draft_config = ModelConfig(**{**c.to_dict(), "n_layers": 1})
    draft = DecoderLM(config=draft_config).to(device=device, dtype=dtype).eval()
    spec, spec_ms = timed(lambda: speculative_generate(model, draft, prompts[0], args.decode_tokens, params=SamplingParams(0)), device)
    baseline = generate(model, prompts[0], args.decode_tokens, SamplingParams(0))
    if spec.tokens != baseline.tokens:
        raise AssertionError("speculative greedy output differs from target")
    report["speculative_smoke"] = {"single_run_ms": spec_ms, "target_calls": spec.target_calls,
        "draft_calls": spec.draft_calls, "baseline_target_calls": baseline.forward_calls,
        "proposed": spec.proposed, "accepted": spec.accepted, "random_draft": True,
        "note": "Single functional run, not a statistically meaningful speed comparison."}
    if args.profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
            batched()
            synchronize(device)
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(args.profile))
    report["notes"] = [
        "Synchronized wall time includes Python dispatch and device execution.",
        "Peak CUDA allocation includes the model, reference logits, pool, and allocator-visible temporaries.",
        "Paged attention is a block-streaming PyTorch implementation, not a fused CUDA kernel.",
        "Engine and speculative runs have different accounting from forced-token decode.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
