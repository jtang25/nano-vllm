"""Focused executed comparison of contiguous reservation and paged KV layouts."""
import argparse
import json
import time
import gc
import torch

from model.checkpoint import load_checkpoint
from benchmarks.experiment_utils import environment, save
from benchmarks.sweep_memory import probe
from benchmarks.workload import load_workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument("--shared-prefix", type=int, default=512)
    parser.add_argument("--kv-gib", type=float, default=4)
    args = parser.parse_args()
    model = load_checkpoint(args.checkpoint, args.device, getattr(torch, args.dtype))
    rows = load_workload(args.workload)[:args.requests]
    lengths = [min(len(row["prompt"]), args.capacity - args.decode_tokens) for row in rows]
    context, budget = max(lengths), int(args.kv_gib * 2**30)
    runs = []
    for layout, size, prefix in [("contiguous", 16, 0)] + [("paged", b, 0) for b in (8, 16, 32, 64)] + [("paged_shared", 16, args.shared_prefix)]:
        start = time.perf_counter()
        result = probe(model, "paged" if layout.startswith("paged") else layout, args.requests, context,
                       args.capacity, budget, size, 128, args.decode_tokens, prefix, lengths)
        runs.append({"layout": layout, "block_size": size, "shared_prefix": prefix,
                     "seconds": time.perf_counter() - start, **result})
        gc.collect()
        torch.cuda.empty_cache()
        if not result["fits"]:
            raise RuntimeError(f"{layout}/{size} did not fit: {result.get('reason')}")
    dense = runs[0]["prefill_snapshot"]["assigned_bytes"]
    paged = next(r for r in runs if r["layout"] == "paged" and r["block_size"] == 16)["prefill_snapshot"]["assigned_bytes"]
    shared = runs[-1]["prefill_snapshot"]["assigned_bytes"]
    report = {"environment": environment(model, args.checkpoint, args.workload), "requests": args.requests,
        "capacity": args.capacity, "lengths": lengths, "runs": runs,
        "paged_16_allocated_reduction_vs_contiguous": 1 - paged / dense,
        "shared_prefix_reduction_vs_unshared_paged_16": 1 - shared / paged,
        "definitions": {"contiguous": "requests * capacity slots", "paged": "unique assigned physical blocks",
                        "internal_waste": "assigned bytes minus unique valid payload, divided by assigned bytes"}}
    save(args.output, report)
    print(json.dumps({k: report[k] for k in ("paged_16_allocated_reduction_vs_contiguous", "shared_prefix_reduction_vs_unshared_paged_16")}, indent=2))


if __name__ == "__main__":
    main()
