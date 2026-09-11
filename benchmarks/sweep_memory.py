"""Measure allocation waste and executable concurrency under an explicit KV budget."""
import argparse
import math
import torch

from benchmarks.benchmark import benchmark
from model.checkpoint import load_checkpoint
from benchmarks.experiment_utils import environment, save, sync
from storage.paged_cache import PagedCache
from scripts.run import make_pool
from benchmarks.workload import load_workload


def accounting(caches, pool=None):
    if pool is None:
        allocated = sum(layer.allocated_bytes for cache in caches for layer in cache)
        live = sum(layer.live_bytes for cache in caches for layer in cache)
        return {"assigned_bytes": allocated, "unique_live_bytes": live, "logical_live_bytes": live,
                "waste_fraction": 1 - live / allocated}
    valid = {}
    for cache in caches:
        for i, block in enumerate(cache.block_table):
            valid[block] = max(valid.get(block, 0), min(pool.block_size, cache.length - i * pool.block_size))
    unique_live = sum(valid.values()) * pool.bytes_per_block // pool.block_size
    assigned = len(valid) * pool.bytes_per_block
    return {"assigned_bytes": assigned, "unique_live_bytes": unique_live,
            "logical_live_bytes": sum(c.live_bytes for c in caches),
            "waste_fraction": 1 - unique_live / assigned,
            "pool_backing_bytes": pool.allocated_bytes, "unique_blocks": len(valid)}


@torch.inference_mode()
def probe(model, layout, count, context, capacity, budget_bytes, block_size, chunk, decode_tokens, shared_prefix=0, lengths=None):
    c = model.config
    lengths = lengths[:count] if lengths is not None else [context] * count
    if len(lengths) != count or shared_prefix > min(lengths):
        raise ValueError("invalid request lengths/shared prefix")
    context = max(lengths)
    per_token = 2 * c.n_layers * c.n_kv_heads * c.head_dim * model.token_embedding.weight.element_size()
    if capacity < context + decode_tokens or capacity > c.max_seq_len:
        raise ValueError("capacity must cover context+decode and fit model maximum")
    if layout == "contiguous" and count * capacity * per_token > budget_bytes:
        return {"fits": False, "reason": "fixed KV budget"}
    caches, pool, source = [], None, None
    try:
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model.device)
        if layout == "paged":
            blocks = budget_bytes // (per_token * block_size)
            needed = sum(math.ceil((length + decode_tokens) / block_size) for length in lengths) - (count - 1) * (shared_prefix // block_size)
            if needed > blocks:
                return {"fits": False, "reason": "paged KV budget"}
            pool = make_pool(model, blocks, block_size)
            if shared_prefix:
                source = PagedCache(pool)
                for start in range(0, shared_prefix, chunk):
                    ids = torch.full((1, min(chunk, shared_prefix - start)), 1, device=model.device, dtype=torch.long)
                    model(ids, caches=source, last_only=True)
                caches = [source.fork() for _ in range(count)]
            else:
                caches = [PagedCache(pool) for _ in range(count)]
        else:
            caches = [model.make_kv_caches(1, capacity) for _ in range(count)]
        start = shared_prefix if layout == "paged" else 0
        for position in range(start, context, chunk):
            eligible = [i for i, length in enumerate(lengths) if length > position]
            chunks = [torch.full((1, min(chunk, lengths[i] - position)), 1, device=model.device, dtype=torch.long) for i in eligible]
            model.forward_batch(chunks, [caches[i] for i in eligible], last_only=True)
        if source:
            source.close()
        snapshot = accounting(caches, pool)
        for _ in range(decode_tokens):
            chunks = [torch.tensor([[2]], device=model.device) for _ in range(count)]
            outputs = model.forward_batch(chunks, caches, last_only=True)
            if not all(bool(torch.isfinite(output).all()) for output in outputs):
                raise AssertionError("nonfinite decode output")
        sync(model.device)
        return {"fits": True, "prefill_snapshot": snapshot, "decode_snapshot": accounting(caches, pool),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(model.device) if model.device.type == "cuda" else None}
    except (torch.OutOfMemoryError, MemoryError) as error:
        return {"fits": False, "reason": type(error).__name__}
    finally:
        for cache in caches:
            if isinstance(cache, PagedCache):
                cache.close()
        if source:
            source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--workload", help="optional saved prompt-length distribution for fragmentation snapshots")
    parser.add_argument("--capacity", type=int, default=8192)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--kv-gib", type=float, default=8)
    parser.add_argument("--max-sequences", type=int, default=128)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--shared-prefix", type=int, default=0)
    parser.add_argument("--timing-repeats", type=int, default=3)
    args = parser.parse_args()
    if not 0 <= args.shared_prefix <= args.context or min(args.context, args.capacity, args.max_sequences, args.chunk) < 1:
        parser.error("invalid memory workload")
    model = load_checkpoint(args.checkpoint, args.device, getattr(torch, args.dtype))
    for layer in model.layers:
        layer.attention.backend = "sdpa"
    runs = []
    lengths = [len(row["prompt"]) for row in load_workload(args.workload)] if args.workload else None
    if lengths is not None and len(lengths) < args.max_sequences:
        parser.error("workload must contain at least max-sequences rows")
    for layout, size in [("contiguous", args.block_sizes[0])] + [("paged", b) for b in args.block_sizes]:
        low, high, trials = 0, args.max_sequences + 1, []
        while high - low > 1:
            count = (high + low) // 2
            result = probe(model, layout, count, args.context, args.capacity, int(args.kv_gib * 2**30), size,
                           args.chunk, args.decode_tokens, args.shared_prefix if layout == "paged" else 0, lengths)
            trials.append({"sequences": count, **result})
            if result["fits"]:
                low = count
            else:
                high = count
            if model.device.type == "cuda":
                torch.cuda.empty_cache()
        runs.append({"layout": layout, "block_size": size, "max_tested_concurrent": low,
                     "search_capped": low == args.max_sequences, "trials": trials})
    timings = {}
    for size in args.block_sizes:
        timings[str(size)] = benchmark(model, args.context, args.decode_tokens, args.timing_repeats, 1, size, ["paged"])
    save(args.output, {"environment": environment(model, args.checkpoint, args.workload), "runs": runs, "block_timings": timings,
        "prompt_lengths": lengths,
        "context": args.context, "capacity": args.capacity, "kv_gib": args.kv_gib, "shared_prefix": args.shared_prefix,
        "decode_tokens": args.decode_tokens,
        "notes": ["Concurrency is an executed prefill plus bounded decode under a fixed KV budget, not a latency-SLO capacity.",
                  "Waste = (assigned physical bytes - unique valid payload bytes) / assigned physical bytes.",
                  "Unassigned pool backing bytes are reported separately, not called fragmentation.",
                  "Fully aligned fixed contexts can have zero paged waste; never manufacture a nonzero percentage."]})


if __name__ == "__main__":
    main()
