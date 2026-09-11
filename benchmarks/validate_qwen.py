"""Compare imported trained weights with Transformers before timing native inference."""
import argparse
import json
import math
import time
import torch

from model.checkpoint import load_checkpoint
from benchmarks.experiment_utils import environment, save
from runtime.generation import generate
from storage.paged_cache import PagedCache
from scripts.run import make_pool
from runtime.sampling import SamplingParams
from runtime.speculative import speculative_generate
from model.tokenizer import check_tokenizers


@torch.inference_mode()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--draft")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-prompts", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = getattr(torch, args.dtype)
    native = load_checkpoint(args.checkpoint, "cuda", dtype)
    reference = AutoModelForCausalLM.from_pretrained(args.hf_model, torch_dtype=dtype, attn_implementation="sdpa").to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    prompts = ["Explain why the sky is blue in two sentences.", "Write a Python function that reverses a list.",
               "Continue this sequence: 2, 4, 6, 8,", "Summarize: " + "A GPU stores model weights and cached attention keys and values in device memory. " * 120]
    cases = []
    precision_check = None
    for index, text in enumerate(prompts[:args.max_prompts]):
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)
        ids = torch.tensor([prompt], device="cuda")
        expected = reference(ids).logits[..., :native.vocab_size]
        actual = native(ids)
        split = min(13, len(prompt) - 1)
        caches = [native.make_kv_caches(1, len(prompt)), PagedCache(make_pool(native, math.ceil(len(prompt) / 16), 16))]
        paths = {"uncached": actual}
        for name, cache in zip(("contiguous", "paged"), caches):
            parts = [native(ids[:, :split], caches=cache), native(ids[:, split:-1], caches=cache), native(ids[:, -1:], caches=cache)]
            paths[name] = torch.cat(parts, dim=1)
            if isinstance(cache, PagedCache):
                cache.close()
        # Q/K row permutation changes dot-product reduction order. Check the
        # source of fp32 drift in double precision before accepting a tolerance.
        if index == 0 and dtype == torch.float32:
            precise_expected = reference.double()(ids).logits[..., :native.vocab_size]
            precise_actual = native.double()(ids)
            precision_check = {"max_abs_diff": (precise_actual - precise_expected).abs().max().item(),
                               "argmax_match": torch.equal(precise_actual.argmax(-1), precise_expected.argmax(-1))}
            reference.float()
            native.float()
        tolerance = .002 if dtype == torch.float32 else .25
        for name, output in paths.items():
            diff = (output.float() - expected.float()).abs()
            p, q = expected.float().log_softmax(-1), output.float().log_softmax(-1)
            kl = (p.exp() * (p - q)).sum(-1).clamp_min(0)
            cases.append({"prompt_length": len(prompt), "path": name, "max_abs_diff": diff.max().item(),
                "mean_abs_diff": diff.mean().item(), "mean_conditional_kl": kl.mean().item(),
                "max_conditional_kl": kl.max().item(), "argmax_match": (expected.argmax(-1) == output.argmax(-1)).sum().item() / expected.shape[1],
                "within_tolerance": bool(torch.allclose(output, expected, atol=tolerance, rtol=tolerance))})
        del paths, expected, actual
    del reference
    torch.cuda.empty_cache()
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompts[0]}], tokenize=True, add_generation_prompt=True)
    start = time.perf_counter()
    normal = generate(native, prompt, 32, SamplingParams(0))
    sample = {"prompt": prompts[0], "text": tokenizer.decode(normal.generated), "target_only_seconds": time.perf_counter() - start}
    if args.draft:
        draft = load_checkpoint(args.draft, "cuda", dtype)
        check_tokenizers(native, draft)
        start = time.perf_counter()
        spec = speculative_generate(native, draft, prompt, 32, params=SamplingParams(0))
        sample.update(speculative_seconds=time.perf_counter() - start, greedy_match=normal.tokens == spec.tokens,
                      accepted=spec.accepted, proposed=spec.proposed, target_calls=spec.target_calls, draft_calls=spec.draft_calls)
    report = {"environment": environment(native, args.checkpoint), "cases": cases, "generation": sample, "double_precision_check": precision_check,
              "note": "FP32 is the strict importer check; BF16 reports numerical differences separately."}
    save(args.output, report)
    print(json.dumps(sample, indent=2))
    if not all(c["within_tolerance"] for c in cases) or sample.get("greedy_match") is False:
        raise SystemExit("parity failed; inspect the saved numerical report")
    if precision_check and (precision_check["max_abs_diff"] > 1e-4 or not precision_check["argmax_match"]):
        raise SystemExit("high-precision importer check failed")


if __name__ == "__main__":
    main()
