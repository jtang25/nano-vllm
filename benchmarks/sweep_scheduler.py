"""Replay a fixed arrival trace under scheduler ablations."""
import argparse
import time
import torch

from model.checkpoint import load_checkpoint
from runtime.engine import Engine
from benchmarks.experiment_utils import environment, save, sync
from scripts.run import make_pool
from runtime.sampling import SamplingParams
from benchmarks.workload import load_workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--budgets", nargs="+", type=int, default=[256, 512, 1024, 2048])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--kv-gib", type=float, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--minimal", action="store_true", help="only unchunked FIFO and chunked decode priority")
    args = parser.parse_args()
    model = load_checkpoint(args.checkpoint, args.device, getattr(torch, args.dtype))
    for layer in model.layers:
        layer.attention.backend = "sdpa"
    c = model.config
    block_bytes = 2 * c.n_layers * c.n_kv_heads * c.head_dim * args.block_size * model.token_embedding.weight.element_size()
    pool = make_pool(model, int(args.kv_gib * 2**30) // block_bytes, args.block_size)
    rows = sorted(load_workload(args.workload), key=lambda row: row.get("arrival_ms", 0))
    policies = [("unchunked_fifo", False, False, max(args.budgets))]
    if not args.minimal:
        policies += [("unchunked_decode_first", False, True, max(args.budgets))]
        policies += [(f"chunked_fifo_{budget}", True, False, budget) for budget in args.budgets]
    policies += [(f"chunked_decode_first_{budget}", True, True, budget) for budget in args.budgets]
    runs = []
    for repeat in range(args.warmup + args.repeats):
        for name, chunked, decode_first, budget in (policies if repeat % 2 == 0 else list(reversed(policies))):
            engine = Engine(model, pool, budget, args.batch, chunked_prefill=chunked, decode_first=decode_first)
            sync(model.device)
            start = time.perf_counter()
            next_row = 0
            try:
                while next_row < len(rows) or engine.waiting or engine.active:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    while next_row < len(rows) and rows[next_row].get("arrival_ms", 0) <= elapsed_ms:
                        row = rows[next_row]
                        request_id = engine.submit(row["prompt"], row["max_new_tokens"], SamplingParams(0))
                        # Queue latency includes time spent waiting for a busy step to return.
                        engine.requests[request_id].submitted_at = start + row.get("arrival_ms", 0) / 1000
                        next_row += 1
                    if engine.active or engine.waiting:
                        engine.step()
                    else:
                        time.sleep(min(.001, max(0, rows[next_row].get("arrival_ms", 0) / 1000 - (time.perf_counter() - start))))
                sync(model.device)
                seconds = time.perf_counter() - start
                if repeat >= args.warmup:
                    metrics = []
                    for request in engine.requests.values():
                        events = request.token_times
                        metrics.append({"ttft_ms": (events[0] - request.submitted_at) * 1000,
                            "tpot_ms": (events[-1] - events[0]) * 1000 / (len(events) - 1) if len(events) > 1 else None,
                            "inter_token_ms": [(b - a) * 1000 for a, b in zip(events, events[1:])],
                            "prompt_length": request.prompt_length, "output_tokens": len(events)})
                    runs.append({"policy": name, "repeat": repeat - args.warmup, "seconds": seconds,
                                 "output_tokens_per_second": sum(r["output_tokens"] for r in metrics) / seconds,
                                 "requests": metrics, "forward_calls": engine.forward_calls})
            finally:
                engine.close()
    save(args.output, {"environment": environment(model, args.checkpoint, args.workload), "runs": runs,
                      "kv_gib": args.kv_gib, "batch": args.batch,
                      "note": "Scheduled arrival timestamps include server-busy delay. Compare identical traces and report TTFT and TPOT together."})


if __name__ == "__main__":
    main()
