"""Separate greedy integration correctness, conditional-logit KL, and resampling statistics."""
import argparse
import json
from pathlib import Path
import torch

from model.checkpoint import load_checkpoint
from benchmarks.experiment_utils import environment, save
from runtime.generation import generate
from runtime.sampling import SamplingParams, residual_distribution
from runtime.speculative import speculative_generate
from benchmarks.workload import load_workload, fingerprint


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--draws", type=int, default=128)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    target = load_checkpoint(args.target, args.device, getattr(torch, args.dtype))
    draft = load_checkpoint(args.draft, args.device, getattr(torch, args.dtype))
    rows = load_workload(args.workload)[args.offset:args.offset + args.limit]
    if len(rows) < args.limit:
        parser.error("workload has fewer prompts than --limit")
    empirical = torch.zeros(args.bins, dtype=torch.float64)
    expected_bins = torch.zeros_like(empirical)
    rng = torch.Generator().manual_seed(args.seed)
    results = []
    progress = Path(args.output).with_suffix(".jsonl")
    progress.parent.mkdir(parents=True, exist_ok=True)
    if progress.exists():
        raise FileExistsError(f"choose a fresh output; preserving {progress}")
    for row in rows:
        prompt = row["prompt"]
        ids = torch.tensor([prompt], device=target.device)
        reference = target(ids, last_only=True)[0, -1].double()
        cache = target.make_kv_caches(1, len(prompt))
        split = max(1, len(prompt) // 2)
        actual = target(ids[:, :split], caches=cache, last_only=True)[0, -1]
        if split < len(prompt):
            actual = target(ids[:, split:], caches=cache, last_only=True)[0, -1]
        p_log, cached_log = reference.log_softmax(-1), actual.double().log_softmax(-1)
        conditional_kl = (p_log.exp() * (p_log - cached_log)).sum().clamp_min(0).item()
        greedy = generate(target, prompt, args.tokens, SamplingParams(0)).tokens
        speculative = speculative_generate(target, draft, prompt, args.tokens, params=SamplingParams(0)).tokens
        p = reference.softmax(-1).cpu()
        q = draft(ids, last_only=True)[0, -1].double().softmax(-1).cpu()
        proposed = torch.multinomial(q, args.draws, replacement=True, generator=rng)
        accepted = torch.rand(args.draws, generator=rng) < (p[proposed] / q[proposed]).clamp(max=1)
        emitted = proposed.clone()
        rejected = int((~accepted).sum())
        if rejected:
            emitted[~accepted] = torch.multinomial(residual_distribution(p, q), rejected, replacement=True, generator=rng)
        empirical += torch.bincount(emitted % args.bins, minlength=args.bins)
        expected_bins.scatter_add_(0, torch.arange(len(p)) % args.bins, p)
        results.append({"id": row["id"], "greedy_sequence_match": greedy == speculative,
                        "target_generated": greedy[len(prompt):], "speculative_generated": speculative[len(prompt):],
                        "matching_generated_tokens": sum(a == b for a, b in zip(greedy[len(prompt):], speculative[len(prompt):])),
                        "conditional_next_token_kl": conditional_kl})
        with progress.open("a", encoding="utf-8") as file:
            file.write(json.dumps(results[-1]) + "\n")
        if len(results) % 25 == 0:
            print(f"Validated {len(results)}/{len(rows)} prompts", flush=True)
    expected_bins /= len(rows)
    empirical /= empirical.sum()
    valid = empirical > 0
    empirical_kl = (empirical[valid] * (empirical[valid] / expected_bins[valid]).log()).sum().item()
    match = sum(r["greedy_sequence_match"] for r in results) / len(results)
    save(args.output, {"environment": environment(target, args.target, args.workload),
        "draft_sha256": fingerprint(args.draft), "prompts": len(rows), "results": results,
        "greedy_sequence_match": match,
        "mean_conditional_next_token_kl": sum(r["conditional_next_token_kl"] for r in results) / len(results),
        "max_conditional_next_token_kl": max(r["conditional_next_token_kl"] for r in results),
        "resampling_binned_empirical_kl": empirical_kl, "bins": args.bins, "draws_per_prompt": args.draws,
        "empirical_bin_probabilities": empirical.tolist(), "expected_bin_probabilities": expected_bins.tolist(),
        "interpretation": "Conditional KL compares cached vs uncached target logits at the same prefixes. Binned empirical KL tests the resampling rule, not the full sequence distribution. Greedy agreement alone is not distribution preservation."})
    if match != 1:
        raise SystemExit("Greedy divergence detected; investigate all mismatches before claiming correctness.")


if __name__ == "__main__":
    main()
