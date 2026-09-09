"""Benchmark n-gram proposals against target-only greedy generation."""
import argparse
import math
import statistics

import torch

from model.checkpoint import load_checkpoint
from runtime.engine import Engine
from benchmarks.experiment_utils import environment, save, timed
from runtime.prompt_lookup import NgramProposer
from scripts.run import make_pool
from runtime.sampling import SamplingParams
from runtime.speculative import speculative_generate_with_proposer
from benchmarks.workload import load_workload


def request_latency(events, starts):
    ttft = [(row[0] - start) * 1000 for row, start in zip(events, starts) if row]
    tpot = [(row[-1] - row[0]) * 1000 / (len(row) - 1) for row in events if len(row) > 1]
    return {"ttft_ms": ttft, "tpot_ms": tpot, "median_ttft_ms": statistics.median(ttft),
            "median_tpot_ms": statistics.median(tpot)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--min-ngrams", nargs="+", type=int, default=[1])
    parser.add_argument("--max-ngrams", nargs="+", type=int, default=[4])
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int, help="use the first N workload rows")
    parser.add_argument("--workload-note", default="Synthetic repetitive continuation workload; do not generalize to ordinary chat prompts.")
    args = parser.parse_args()
    model = load_checkpoint(args.checkpoint, args.device, getattr(torch, args.dtype))
    rows = load_workload(args.workload)[:args.limit]
    if not rows:
        parser.error("workload selection is empty")
    configurations = [(minimum, maximum) for minimum in args.min_ngrams for maximum in args.max_ngrams
                      if 1 <= minimum <= maximum]
    if not configurations:
        parser.error("every n-gram configuration must satisfy 1 <= min <= max")
    params = SamplingParams(0)

    def pool_for(row):
        blocks = math.ceil((len(row["prompt"]) + row["max_new_tokens"]) / args.block_size)
        return make_pool(model, blocks, args.block_size)

    def baseline(row):
        pool = pool_for(row)
        engine = Engine(model, pool, len(row["prompt"]), 1)
        try:
            engine.submit(row["prompt"], row["max_new_tokens"], params)
            request = next(iter(engine.run().values()))
            return {"tokens": request.tokens, "events": request.token_times, "calls": engine.forward_calls}
        finally:
            engine.close()

    report = {"environment": environment(model, args.checkpoint, args.workload), "requests": len(rows),
              "lookahead": args.lookahead, "warmup": args.warmup, "runs": [],
              "match_selection": "earliest occurrence in the original sequence among longest matches",
              "verification": "variable proposal lengths are packed; rollback is per sequence",
              "scope": "offline batching, one device; no scheduler integration, Numba, or GPU lookup kernel",
              "note": args.workload_note}

    for minimum, maximum in configurations:
        samples = []
        for repeat in range(args.warmup + args.repeats):
            baseline_outputs, lookup_outputs = [], []
            baseline_starts, lookup_starts = [], []
            baseline_seconds = lookup_seconds = 0
            match_sizes = []
            order = ("baseline", "lookup") if repeat % 2 == 0 else ("lookup", "baseline")
            for mode in order:
                for row in rows:
                    if mode == "baseline":
                        result, start, elapsed = timed(lambda row=row: baseline(row), model.device)
                        baseline_outputs.append(result)
                        baseline_starts.append(start)
                        baseline_seconds += elapsed
                    else:
                        pool = pool_for(row)
                        proposer = NgramProposer(minimum, maximum)
                        result, start, elapsed = timed(
                            lambda row=row, pool=pool, proposer=proposer: speculative_generate_with_proposer(
                                model, proposer, row["prompt"], row["max_new_tokens"], args.lookahead,
                                params, target_pool=pool), model.device)
                        lookup_outputs.append(result)
                        lookup_starts.append(start)
                        lookup_seconds += elapsed
                        match_sizes.extend(match["ngram_size"] for match in proposer.matches)
            if repeat < args.warmup:
                continue
            baseline_events = [output["events"] for output in baseline_outputs]
            lookup_events = [output.token_times for output in lookup_outputs]
            samples.append({"baseline_seconds": baseline_seconds, "lookup_seconds": lookup_seconds,
                "baseline_latency": request_latency(baseline_events, baseline_starts),
                "lookup_latency": request_latency(lookup_events, lookup_starts),
                "baseline_target_calls": sum(output["calls"] for output in baseline_outputs),
                "lookup_target_calls": sum(output.target_calls for output in lookup_outputs),
                "proposed": sum(output.proposed for output in lookup_outputs),
                "accepted": sum(output.accepted for output in lookup_outputs),
                "bonus_tokens": sum(output.bonus_tokens for output in lookup_outputs),
                "lookup_rounds": sum(len(output.accepted_per_round) for output in lookup_outputs),
                "matched_ngram_sizes": match_sizes,
                "greedy_sequence_match": sum(a["tokens"] == b.tokens for a, b in
                                             zip(baseline_outputs, lookup_outputs)) / len(rows)})
        run = {"min_ngram": minimum, "max_ngram": maximum, "samples": samples,
               "tpot_speedup": statistics.median(s["baseline_latency"]["median_tpot_ms"] for s in samples) /
                               statistics.median(s["lookup_latency"]["median_tpot_ms"] for s in samples),
               "end_to_end_speedup": statistics.median(s["baseline_seconds"] for s in samples) /
                                     statistics.median(s["lookup_seconds"] for s in samples)}
        report["runs"].append(run)
    save(args.output, report)


if __name__ == "__main__":
    main()
