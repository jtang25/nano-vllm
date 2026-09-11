"""Run native OR vLLM in separate processes on identical exported weights and IDs."""
import argparse
import json
import statistics
import time
from pathlib import Path
import platform
import os

from benchmarks.workload import fingerprint, load_workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("native", "vllm"), required=True)
    parser.add_argument("--checkpoint", help="native checkpoint for --runtime native")
    parser.add_argument("--model", type=Path, required=True, help="identical Llama export directory")
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-batch-tokens", type=int, default=4096)
    parser.add_argument("--kv-gib", type=float, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--eager", action="store_true", help="disable vLLM graphs for an additional ablation")
    args = parser.parse_args()
    if min(args.batch, args.max_batch_tokens, args.block_size, args.repeats) < 1 or args.kv_gib <= 0 or args.warmup < 0:
        parser.error("sizes/repeats must be positive and warmup nonnegative")
    import torch
    rows = load_workload(args.workload)
    if any(row.get("arrival_ms", 0) for row in rows):
        parser.error("offline throughput comparison requires simultaneous arrivals")
    if len({r["max_new_tokens"] for r in rows}) != 1:
        parser.error("this comparator requires identical output budgets")
    manifest = json.loads((args.model / "manifest.json").read_text())
    if fingerprint(args.model / "model.safetensors") != manifest["weight_sha256"]:
        parser.error("exported weights differ from the manifest")
    if manifest.get("config_sha256") and fingerprint(args.model / "config.json") != manifest["config_sha256"]:
        parser.error("model config differs from the import manifest")
    identity = {"weight_sha256": manifest["weight_sha256"], "workload_sha256": fingerprint(args.workload),
                "batch": args.batch, "max_batch_tokens": args.max_batch_tokens, "dtype": args.dtype,
                "kv_gib": args.kv_gib, "block_size": args.block_size, "prefix_caching": False, "ignore_eos": True}
    versions = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda}
    if args.runtime == "native":
        from model.checkpoint import load_checkpoint
        from runtime.engine import Engine
        from scripts.run import make_pool
        from runtime.sampling import SamplingParams
        if not args.checkpoint or fingerprint(args.checkpoint) != manifest.get("native_checkpoint_sha256"):
            parser.error("native checkpoint hash must match export manifest; export with --native-output")
        model = load_checkpoint(args.checkpoint, args.device, getattr(torch, args.dtype))
        for layer in model.layers:
            layer.attention.backend = "sdpa"
        c = model.config
        block_bytes = 2 * c.n_layers * c.n_kv_heads * c.head_dim * args.block_size * model.token_embedding.weight.element_size()
        pool = make_pool(model, int(args.kv_gib * 2**30) // block_bytes, args.block_size)

        def run():
            engine = Engine(model, pool, args.max_batch_tokens, args.batch, decode_first=True)
            try:
                for row in rows:
                    engine.submit(row["prompt"], row["max_new_tokens"], SamplingParams(0))
                engine.run()
                return [r.generated for r in engine.requests.values()]
            finally:
                engine.close()
    else:
        import vllm
        from vllm import LLM, SamplingParams
        versions["vllm"] = vllm.__version__
        llm = LLM(model=str(args.model), skip_tokenizer_init=True, dtype=args.dtype, tensor_parallel_size=1,
                   max_num_seqs=args.batch, max_num_batched_tokens=args.max_batch_tokens,
                   kv_cache_memory_bytes=int(args.kv_gib * 2**30), enable_prefix_caching=False,
                   enforce_eager=args.eager, seed=42, block_size=args.block_size,
                   max_model_len=max(len(r["prompt"]) + r["max_new_tokens"] for r in rows))
        params = SamplingParams(temperature=0, max_tokens=rows[0]["max_new_tokens"], ignore_eos=True, detokenize=False)

        def run():
            outputs = llm.generate([{"prompt_token_ids": r["prompt"]} for r in rows], params, use_tqdm=False)
            return [list(output.outputs[0].token_ids) for output in outputs]

    def sync():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    samples = []
    for i in range(args.warmup + args.repeats):
        sync()
        start = time.perf_counter()
        outputs = run()
        sync()
        elapsed = time.perf_counter() - start
        if len(outputs) != len(rows) or any(len(output) != row["max_new_tokens"] for output, row in zip(outputs, rows)):
            raise AssertionError("runtime emitted an unexpected number of tokens")
        if i >= args.warmup:
            samples.append(elapsed)
    report = {"runtime": args.runtime, "identity": identity, "versions": versions, "manifest": manifest,
              "warmup": args.warmup, "seconds": samples, "median_seconds": statistics.median(samples),
              "output_tokens_per_second": sum(map(len, outputs)) / statistics.median(samples),
              "outputs": outputs, "vllm_eager": args.eager if args.runtime == "vllm" else None,
              "attention_tile_tokens": int(os.environ.get("NANOVLLM_ATTENTION_TILE_TOKENS", "1024")) if args.runtime == "native" else None,
              "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("outputs", "manifest")}, indent=2))


if __name__ == "__main__":
    main()
