"""Summarize raw runs without substituting desired resume numbers."""
import argparse
import json
from pathlib import Path
import statistics
import random


def median(values):
    return statistics.median(values)


def paired_ratio_ci(a, b, seed=42):
    if len(a) != len(b) or not a:
        raise ValueError("paired measurements required")
    if len(a) < 3:
        return None
    rng = random.Random(seed)
    ratios = []
    for _ in range(2000):
        indices = [rng.randrange(len(a)) for _ in a]
        ratios.append(median([a[i] for i in indices]) / median([b[i] for i in indices]))
    ratios.sort()
    return [ratios[50], ratios[1949]]


def summarize_scheduler(report):
    groups = {}
    for run in report["runs"]:
        groups.setdefault(run["policy"], []).append(run)
    base = sorted(groups["unchunked_fifo"], key=lambda r: r["repeat"])
    baseline_ttft = [median(r["ttft_ms"] for r in run["requests"]) for run in base]
    baseline_tpot = [median(r["tpot_ms"] for r in run["requests"] if r["tpot_ms"] is not None) for run in base]
    output = {}
    for name, runs in groups.items():
        runs = sorted(runs, key=lambda r: r["repeat"])
        ttft = [median(r["ttft_ms"] for r in run["requests"]) for run in runs]
        tpot = [median(r["tpot_ms"] for r in run["requests"] if r["tpot_ms"] is not None) for run in runs]
        output[name] = {"p50_ttft_ms": median(ttft), "p50_tpot_ms": median(tpot),
            "ttft_speedup_vs_unchunked": median(baseline_ttft) / median(ttft),
            "ttft_speedup_bootstrap_95": paired_ratio_ci(baseline_ttft, ttft),
            "tpot_ratio_vs_unchunked": median(tpot) / median(baseline_tpot),
            "tpot_ratio_bootstrap_95": paired_ratio_ci(tpot, baseline_tpot)}
    return output


def summarize_speculation(report):
    output = []
    for run in report["runs"]:
        samples = run["samples"]
        baseline = [s["baseline_latency"]["median_tpot_ms"] for s in samples]
        spec = [s["speculative_latency"]["median_tpot_ms"] for s in samples]
        output.append({"batch": run["batch"], "baseline_tpot_ms": median(baseline),
            "speculative_amortized_decode_ms_per_token": median(s["speculative_amortized_decode_ms_per_token"] for s in samples),
            "speculative_tpot_ms": median(spec), "tpot_speedup": median(baseline) / median(spec),
            "tpot_speedup_bootstrap_95": paired_ratio_ci(baseline, spec),
            "target_call_reduction": median(s["forward_call_reduction"] for s in samples),
            "emitted_tokens_per_sequence_verification": median(s["emitted_tokens_per_sequence_verification"] for s in samples),
            "accepted_draft_fraction": sum(s["accepted"] for s in samples) / sum(s["proposed"] for s in samples)})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler")
    parser.add_argument("--speculation")
    parser.add_argument("--memory")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = {}
    if args.scheduler:
        report["scheduler"] = summarize_scheduler(json.loads(Path(args.scheduler).read_text()))
    if args.speculation:
        report["speculation"] = summarize_speculation(json.loads(Path(args.speculation).read_text()))
    if args.memory:
        memory = json.loads(Path(args.memory).read_text())
        report["memory"] = []
        for run in memory["runs"]:
            fits = [r for r in run["trials"] if r["fits"]]
            best = max(fits, key=lambda r: r["sequences"]) if fits else None
            report["memory"].append({"layout": run["layout"], "block_size": run["block_size"],
                "concurrency": run["max_tested_concurrent"], "lower_bound_only": run["search_capped"],
                "snapshot": best["prefill_snapshot"] if best else None})
    report["note"] = "Bootstrap intervals resample paired runs, not tokens. One repetition cannot establish uncertainty."
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
