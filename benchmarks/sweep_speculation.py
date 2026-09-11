"""Measure actual batched speculative decoding against the target-only engine."""
import argparse
import math
import statistics
import torch

from model.checkpoint import load_checkpoint
from runtime.engine import Engine
from benchmarks.experiment_utils import environment, latency, save, timed
from scripts.run import make_pool
from runtime.sampling import SamplingParams
from runtime.speculative_batch import speculative_batch
from benchmarks.workload import load_workload, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(4)
    target = load_checkpoint(args.target, args.device, getattr(torch, args.dtype))
    draft = load_checkpoint(args.draft, args.device, getattr(torch, args.dtype))
    for model in (target, draft):
        for layer in model.layers:
            layer.attention.backend = "sdpa"
    rows = load_workload(args.workload)
    params = SamplingParams(args.temperature)
    report = {"environment": environment(target, args.target, args.workload),
              "draft_sha256": fingerprint(args.draft), "lookahead": args.lookahead,
              "temperature": args.temperature, "warmup": args.warmup, "runs": []}
    for batch in args.batches:
        if not 1 <= batch <= len(rows):
            parser.error("batch must be positive and no larger than the workload")
        selected = rows[:batch]
        if len({row["max_new_tokens"] for row in selected}) != 1:
            parser.error("speculative sweep requires equal output budgets within a batch")
        prompts = [row["prompt"] for row in selected]
        tokens = selected[0]["max_new_tokens"]
        if tokens <= args.lookahead + 1:
            parser.error("use more than lookahead+1 tokens so TPOT covers more than the first burst")
        blocks = sum(math.ceil((len(p) + tokens) / args.block_size) for p in prompts)
        tp, dp = make_pool(target, blocks, args.block_size), make_pool(draft, blocks, args.block_size)

        def baseline(seed):
            engine = Engine(target, tp, sum(map(len, prompts)), batch)
            try:
                for i, prompt in enumerate(prompts):
                    engine.submit(prompt, tokens, params, seed=seed + i)
                engine.run()
                return {"tokens": [r.tokens for r in engine.requests.values()],
                        "events": [r.token_times for r in engine.requests.values()], "calls": engine.forward_calls}
            finally:
                engine.close()

        samples = []
        for repeat in range(args.warmup + args.repeats):
            functions = {
                "baseline": lambda: baseline(42 + repeat),
                "speculative": lambda: speculative_batch(target, draft, prompts, tokens, args.lookahead, params,
                                    seed=42 + repeat, target_pool=tp, draft_pool=dp),
            }
            measured = {name: timed(functions[name], target.device)
                        for name in (("baseline", "speculative") if repeat % 2 == 0 else ("speculative", "baseline"))}
            base, base_start, base_seconds = measured["baseline"]
            spec, spec_start, spec_seconds = measured["speculative"]
            sequence_matches = [a == b for a, b in zip(base["tokens"], spec.tokens)]
            if repeat < args.warmup:
                continue
            baseline_latency = latency(base["events"], base_start)
            spec_latency = latency(spec.token_times, spec_start)
            samples.append({"baseline_seconds": base_seconds, "speculative_seconds": spec_seconds,
                "speculative_amortized_decode_ms_per_token": statistics.median(
                    (events[-1] - spec.prefill_done_at) * 1000 / len(events) for events in spec.token_times),
                "baseline_latency": baseline_latency, "speculative_latency": spec_latency,
                "baseline_target_calls": base["calls"], "speculative_target_calls": spec.target_calls,
                "draft_calls": spec.draft_calls, "accepted": spec.accepted, "proposed": spec.proposed,
                "greedy_sequence_match": sum(sequence_matches) / len(sequence_matches),
                "forward_call_reduction": 1 - spec.target_calls / base["calls"],
                "verification_calls": spec.target_calls - 1,
                "emitted_tokens_per_target_call": sum(map(len, spec.generated)) / spec.target_calls,
                "emitted_tokens_per_sequence_verification": sum(map(len, spec.generated)) / sum(len(r) for r in spec.rounds),
                "rounds": spec.rounds})
        report["runs"].append({"batch": batch, "samples": samples,
            "tpot_speedup": statistics.median(s["baseline_latency"]["median_tpot_ms"] for s in samples) /
                            statistics.median(s["speculative_latency"]["median_tpot_ms"] for s in samples),
            "end_to_end_speedup": statistics.median(s["baseline_seconds"] for s in samples) /
                                  statistics.median(s["speculative_seconds"] for s in samples)})
        tp, dp = None, None
    report["notes"] = ["Target and draft phases are sequential; this implementation does not hide draft latency by overlap.",
                       "Speculative tokens arrive in bursts. TPOT is elapsed first-to-last delivery divided by N-1.",
                       "Inspect raw samples and confidence intervals; no crossover batch is assumed."]
    save(args.output, report)


if __name__ == "__main__":
    main()
