"""Measure prompt lookup on a GPU, then replay the measured token timeline."""
import argparse
import math
from pathlib import Path
import sys
import time

import torch

from model.checkpoint import load_checkpoint
from runtime.engine import Engine
from benchmarks.experiment_utils import environment, save, timed
from runtime.prompt_lookup import NgramProposer
from scripts.run import make_pool
from runtime.sampling import SamplingParams
from runtime.speculative import speculative_generate_with_proposer
from model.tokenizer import tokenizer_for
from benchmarks.workload import load_workload


GREEN = "\033[38;5;42m"
PINK = "\033[38;5;213m"
DIM = "\033[2m"
RESET = "\033[0m"


def tpot_ms(token_times):
    if len(token_times) < 2:
        return None
    return (token_times[-1] - token_times[0]) * 1000 / (len(token_times) - 1)


def make_request_pool(model, prompt, tokens, block_size):
    blocks = math.ceil((len(prompt) + tokens) / block_size)
    return make_pool(model, blocks, block_size)


def run_baseline(model, prompt, tokens, params, block_size, seed):
    pool = make_request_pool(model, prompt, tokens, block_size)
    engine = Engine(model, pool, max_batch_tokens=max(1, len(prompt)), max_requests=1)
    try:
        engine.submit(prompt, tokens, params, seed=seed)
        request = next(iter(engine.run().values()))
        return request.tokens, request.generated, request.token_times, engine.forward_calls
    finally:
        engine.close()


def run_lookup(model, prompt, tokens, params, block_size, lookahead, min_ngram, max_ngram, seed):
    pool = make_request_pool(model, prompt, tokens, block_size)
    proposer = NgramProposer(min_ngram, max_ngram)
    return speculative_generate_with_proposer(model, proposer, prompt, tokens, lookahead, params,
                                              seed=seed, target_pool=pool)


def render(device_name, model_name, decoded, emitted, total, baseline_tpot, live_tpot, lookup, elapsed,
           finished=False):
    acceptance = lookup.accepted / lookup.proposed if lookup.proposed else 0
    rate = 0 if live_tpot is None else 1000 / live_tpot
    speedup = 0 if live_tpot is None else baseline_tpot / live_tpot
    status = "COMPLETE" if finished else "GENERATING"
    transcript = decoded[-700:].replace("\r", "")
    lines = [
        f"{DIM}recorded {device_name} execution replay · metrics are from the run shown{RESET}",
        f"{PINK}nano-vllm{RESET}  {model_name}  {GREEN}{status}{RESET}",
        "",
        transcript,
        "",
        f"{GREEN}{'━' * max(1, int(48 * emitted / total))}{DIM}{'━' * max(0, 48 - int(48 * emitted / total))}{RESET}",
        f"tokens              {emitted:>4}/{total:<4}",
        f"acceptance          {lookup.accepted}/{lookup.proposed}  ({acceptance:.1%})  {DIM}run total{RESET}",
        f"target calls        {lookup.target_calls:<4}       {DIM}baseline {total}{RESET}",
        f"observed TPOT       {'--' if live_tpot is None else f'{live_tpot:.2f} ms'}",
        f"decode rate         {'--' if live_tpot is None else f'{rate:.1f} tok/s'}",
        f"TPOT speedup        {'--' if live_tpot is None else f'{speedup:.2f}x'}",
        f"replay elapsed      {elapsed:.2f} s",
    ]
    sys.stdout.write("\033[2J\033[H" + "\n".join(lines) + "\n")
    sys.stdout.flush()


def replay(tokenizer, generated, token_times, baseline_tpot, lookup, device_name, model_name, speed):
    start = time.perf_counter()
    for index in range(len(generated)):
        if index:
            delay = max(0, token_times[index] - token_times[index - 1]) / speed
            time.sleep(delay)
        live = tpot_ms(token_times[:index + 1])
        render(device_name, model_name, tokenizer.decode(generated[:index + 1]), index + 1, len(generated),
               baseline_tpot, live, lookup, time.perf_counter() - start, index + 1 == len(generated))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--workload")
    source.add_argument("--prompt")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--min-ngram", type=int, default=1)
    parser.add_argument("--max-ngram", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay-speed", type=float, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.replay_speed <= 0 or args.warmup < 0:
        parser.error("replay speed must be positive and warmup nonnegative")

    model = load_checkpoint(args.checkpoint, args.device, getattr(torch, args.dtype))
    tokenizer = tokenizer_for(model)
    if args.workload:
        rows = load_workload(args.workload)
        if not 0 <= args.index < len(rows):
            parser.error("workload index is out of range")
        row = rows[args.index]
        prompt = row["prompt"]
        tokens = args.tokens or row["max_new_tokens"]
    else:
        prompt = tokenizer.encode(args.prompt)
        tokens = args.tokens or 64
    params = SamplingParams(0)

    print("Warming up and measuring the baseline and prompt-lookup paths...", flush=True)
    for warmup in range(args.warmup):
        run_baseline(model, prompt, tokens, params, args.block_size, args.seed + warmup)
        run_lookup(model, prompt, tokens, params, args.block_size, args.lookahead,
                   args.min_ngram, args.max_ngram, args.seed + warmup)

    baseline, _, baseline_seconds = timed(
        lambda: run_baseline(model, prompt, tokens, params, args.block_size, args.seed), model.device)
    lookup, _, lookup_seconds = timed(
        lambda: run_lookup(model, prompt, tokens, params, args.block_size, args.lookahead,
                           args.min_ngram, args.max_ngram, args.seed), model.device)
    baseline_tokens, baseline_generated, baseline_times, baseline_calls = baseline
    if baseline_tokens != lookup.tokens:
        raise RuntimeError("prompt lookup did not match target-only greedy generation")
    baseline_tpot = tpot_ms(baseline_times)
    lookup_tpot = tpot_ms(lookup.token_times)
    summary = {
        "environment": environment(model, args.checkpoint, args.workload),
        "workload_index": args.index if args.workload else None,
        "prompt_tokens": len(prompt),
        "output_tokens": len(lookup.generated),
        "lookahead": args.lookahead,
        "min_ngram": args.min_ngram,
        "max_ngram": args.max_ngram,
        "baseline": {"seconds": baseline_seconds, "tpot_ms": baseline_tpot,
                     "target_calls": baseline_calls, "tokens_per_second": len(baseline_generated) / baseline_seconds},
        "lookup": {"seconds": lookup_seconds, "tpot_ms": lookup_tpot, "target_calls": lookup.target_calls,
                   "proposed": lookup.proposed, "accepted": lookup.accepted, "bonus_tokens": lookup.bonus_tokens,
                   "tokens_per_second": len(lookup.generated) / lookup_seconds},
        "tpot_speedup": baseline_tpot / lookup_tpot,
        "end_to_end_speedup": baseline_seconds / lookup_seconds,
        "greedy_sequence_match": True,
    }
    if args.output:
        save(args.output, summary)
    model_name = Path(args.checkpoint).stem
    device_name = torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else "CPU"
    replay(tokenizer, lookup.generated, lookup.token_times, baseline_tpot, lookup,
           device_name, model_name, args.replay_speed)
    print(f"\n{DIM}baseline {baseline_tpot:.2f} ms TPOT · lookup {lookup_tpot:.2f} ms · "
          f"{summary['tpot_speedup']:.2f}x · measurements saved "
          f"{'to ' + str(args.output) if args.output else 'only in this terminal'}{RESET}")


if __name__ == "__main__":
    main()
