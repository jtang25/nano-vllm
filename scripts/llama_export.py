"""Export the identical dense model as Llama weights for independent baselines."""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import torch

from model.checkpoint import load_checkpoint, save_checkpoint
from model.config import ModelConfig
from model.layers import DecoderLM


def rope_rows(tensor, heads):
    # Native RoPE uses adjacent pairs; Llama uses split halves inside each head.
    head_dim = tensor.shape[0] // heads
    shape = (heads, head_dim // 2, 2) + tensor.shape[1:]
    return tensor.reshape(shape).transpose(1, 2).reshape_as(tensor).contiguous()


def export_llama(model, directory):
    from safetensors.torch import save_file
    c = model.config
    if c.n_experts:
        raise ValueError("Llama export supports dense models only; do not compare a different MoE architecture")
    if not (c.qkv_bias and c.output_bias and c.mlp_bias):
        raise ValueError("this exporter expects the original biased native architecture; compare imported Qwen with its original HF directory")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        "model.embed_tokens.weight": model.token_embedding.weight.detach().cpu().contiguous(),
        "model.norm.weight": model.final_norm.weight.detach().cpu().contiguous(),
        "lm_head.weight": model.lm_head.weight.detach().cpu().clone().contiguous(),
    }
    for i, layer in enumerate(model.layers):
        prefix = f"model.layers.{i}"
        attn = layer.attention
        for suffix in ("weight", "bias"):
            state[f"{prefix}.self_attn.q_proj.{suffix}"] = rope_rows(getattr(attn.Wq, suffix).detach().cpu(), c.n_q_heads)
            k, v = getattr(attn.Wkv, suffix).detach().cpu().chunk(2, dim=0)
            state[f"{prefix}.self_attn.k_proj.{suffix}"] = rope_rows(k, c.n_kv_heads)
            state[f"{prefix}.self_attn.v_proj.{suffix}"] = v.contiguous()
            state[f"{prefix}.self_attn.o_proj.{suffix}"] = getattr(attn.Wo, suffix).detach().cpu().contiguous()
            for projection in ("gate_proj", "up_proj", "down_proj"):
                state[f"{prefix}.mlp.{projection}.{suffix}"] = getattr(getattr(layer.ffn, projection), suffix).detach().cpu().contiguous()
        state[f"{prefix}.input_layernorm.weight"] = layer.attention_norm.weight.detach().cpu().contiguous()
        state[f"{prefix}.post_attention_layernorm.weight"] = layer.ffn_norm.weight.detach().cpu().contiguous()
    config = dict(architectures=["LlamaForCausalLM"], model_type="llama", vocab_size=c.vocab_size,
        hidden_size=c.d_model, intermediate_size=c.d_ff, num_hidden_layers=c.n_layers,
        num_attention_heads=c.n_q_heads, num_key_value_heads=c.n_kv_heads,
        max_position_embeddings=c.max_seq_len, rms_norm_eps=c.norm_eps, rope_theta=c.rope_base,
        hidden_act="silu", attention_bias=True, mlp_bias=True, tie_word_embeddings=c.tie_embeddings,
        torch_dtype=str(model.dtype).split(".")[-1], bos_token_id=None, eos_token_id=None,
        rope_parameters={"rope_type": "default", "rope_theta": c.rope_base})
    (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    save_file(state, directory / "model.safetensors", metadata={"format": "pt"})
    digest = hashlib.sha256()
    with (directory / "model.safetensors").open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    manifest = {"parameters": sum(p.numel() for p in model.parameters()), "weight_sha256": digest.hexdigest(),
                "native_config": c.to_dict(), "dtype": str(model.dtype), "format": "llama-dense"}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--preset", choices=("tiny", "1.3b"), default="tiny")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, help="override preset depth, e.g. 4 for a smaller draft")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    if args.checkpoint:
        model = load_checkpoint(args.checkpoint, dtype=dtype)
        synthetic = torch.load(args.checkpoint, map_location="cpu", weights_only=True).get("metadata", {}).get("synthetic_weights")
    else:
        synthetic = True
        c = ModelConfig() if args.preset == "tiny" else ModelConfig(vocab_size=32000, d_model=2048,
            n_layers=26, n_q_heads=16, n_kv_heads=4, d_ff=5632, max_seq_len=8192, attention_backend="sdpa")
        if args.layers is not None:
            c = replace(c, n_layers=args.layers)
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            model = DecoderLM(config=c).eval()
        finally:
            torch.set_default_dtype(previous_dtype)
    if args.native_output:
        save_checkpoint(model, args.native_output, {"seed": args.seed, "synthetic_weights": synthetic})
    manifest = export_llama(model, args.output)
    manifest["synthetic_weights"] = synthetic
    from benchmarks.workload import fingerprint
    if args.native_output or args.checkpoint:
        manifest["native_checkpoint_sha256"] = fingerprint(args.native_output or args.checkpoint)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
