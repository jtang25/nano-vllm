"""Import dense Qwen2 weights into our decoder; inference stays in native code."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

from model.checkpoint import save_checkpoint
from model.config import ModelConfig
from model.layers import DecoderLM
from benchmarks.workload import fingerprint


def adjacent_rope_rows(weight, heads):
    width = weight.shape[0] // heads
    return weight.reshape(heads, 2, width // 2, *weight.shape[1:]).transpose(1, 2).reshape_as(weight).contiguous()


def native_config(c, vocab_size=None):
    if c.get("model_type") != "qwen2" or c.get("use_sliding_window") or c.get("rope_scaling"):
        raise ValueError("only dense Qwen2 with ordinary RoPE and full attention is supported")
    if c.get("hidden_act") != "silu":
        raise ValueError("expected SwiGLU")
    vocab_size = c["vocab_size"] if vocab_size is None else vocab_size
    if not 1 <= vocab_size <= c["vocab_size"]:
        raise ValueError("vocabulary override must only remove padded rows")
    return ModelConfig(vocab_size=vocab_size, d_model=c["hidden_size"], n_layers=c["num_hidden_layers"],
        n_q_heads=c["num_attention_heads"], n_kv_heads=c["num_key_value_heads"], d_ff=c["intermediate_size"],
        max_seq_len=c["max_position_embeddings"], norm_eps=c["rms_norm_eps"], rope_base=c["rope_theta"],
        tie_embeddings=c.get("tie_word_embeddings", False), attention_backend="sdpa",
        qkv_bias=True, output_bias=False, mlp_bias=False)


def convert_state(state, config):
    consumed = set()

    def take(name):
        consumed.add(name)
        return state[name]

    result = {"token_embedding.weight": take("model.embed_tokens.weight")[:config.vocab_size],
              "final_norm.weight": take("model.norm.weight")}
    result["lm_head.weight"] = (take("lm_head.weight")[:config.vocab_size] if "lm_head.weight" in state
                                else result["token_embedding.weight"])
    if config.tie_embeddings and not torch.equal(result["lm_head.weight"], result["token_embedding.weight"]):
        raise ValueError("tied embedding weights disagree")
    if not config.tie_embeddings and "lm_head.weight" not in state:
        raise ValueError("untied model is missing lm_head")
    for i in range(config.n_layers):
        hf, native = f"model.layers.{i}", f"layers.{i}"
        for suffix in ("weight", "bias"):
            result[f"{native}.attention.Wq.{suffix}"] = adjacent_rope_rows(take(f"{hf}.self_attn.q_proj.{suffix}"), config.n_q_heads)
            k = adjacent_rope_rows(take(f"{hf}.self_attn.k_proj.{suffix}"), config.n_kv_heads)
            v = take(f"{hf}.self_attn.v_proj.{suffix}")
            result[f"{native}.attention.Wkv.{suffix}"] = torch.cat((k, v), dim=0)
        result[f"{native}.attention.Wo.weight"] = take(f"{hf}.self_attn.o_proj.weight")
        for projection in ("gate_proj", "up_proj", "down_proj"):
            result[f"{native}.ffn.{projection}.weight"] = take(f"{hf}.mlp.{projection}.weight")
        result[f"{native}.attention_norm.weight"] = take(f"{hf}.input_layernorm.weight")
        result[f"{native}.ffn_norm.weight"] = take(f"{hf}.post_attention_layernorm.weight")
    if set(state) - consumed:
        raise ValueError(f"unrecognized checkpoint tensors: {sorted(set(state) - consumed)}")
    return result


def from_state(state, config):
    with torch.device("meta"):
        model = DecoderLM(config=config)
    model.load_state_dict(convert_state(state, config), strict=True, assign=True)
    if config.tie_embeddings:
        model.lm_head.weight = model.token_embedding.weight
    for layer in model.layers:
        width = config.head_dim
        layer.attention.inv_freq = config.rope_base ** (-torch.arange(0, width, 2, dtype=torch.float32) / width)
    return model.eval()


def main():
    from huggingface_hub import HfApi, snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--hf-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, help="trim unused padded vocabulary rows for draft compatibility")
    args = parser.parse_args()
    torch.set_num_threads(4)
    revision = HfApi().model_info(args.model, revision=args.revision).sha
    snapshot_download(args.model, revision=revision, local_dir=args.hf_output,
                      allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json", "LICENSE*", "README.md"])
    tokenizer = AutoTokenizer.from_pretrained(args.hf_output, trust_remote_code=False)
    if args.vocab_size is not None and max(tokenizer.get_vocab().values()) >= args.vocab_size:
        raise ValueError("vocabulary override removes a tokenizer-reachable token")
    config = native_config(json.loads((args.hf_output / "config.json").read_text()), args.vocab_size)
    state = {}
    files = sorted(args.hf_output.glob("*.safetensors"))
    for path in files:
        tensors = load_file(path)
        if set(state) & set(tensors):
            raise ValueError("duplicate tensors across shards")
        state.update(tensors)
    model = from_state(state, config)
    model.tokenizer_spec = {"type": "huggingface", "repo_id": args.model, "revision": revision,
        "backend_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()}
    weights = {path.name: fingerprint(path) for path in files}
    metadata = {"source_model": args.model, "source_revision": revision, "source_weights": weights, "synthetic_weights": False}
    save_checkpoint(model, args.output, metadata)
    manifest = {"parameters": sum(p.numel() for p in model.parameters()), "native_config": config.to_dict(),
        "dtype": str(model.dtype), "format": "qwen2", **metadata, "tokenizer": model.tokenizer_spec,
        "native_checkpoint_sha256": fingerprint(args.output), "config_sha256": fingerprint(args.hf_output / "config.json")}
    # The two selected models each have one safetensors file; comparator checks it.
    if len(files) == 1 and files[0].name == "model.safetensors":
        manifest["weight_sha256"] = weights["model.safetensors"]
    (args.hf_output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
