"""Pin a public evaluation split and save identical Qwen token IDs for all runners."""
import argparse
import json
import random
from pathlib import Path

from benchmarks.workload import fingerprint


def main():
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, default=Path("workloads/qwen"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    repo = "HuggingFaceH4/no_robots"
    api = HfApi()
    revision = api.dataset_info(repo).sha
    files = api.list_repo_files(repo, repo_type="dataset", revision=revision)
    file = next(f for f in files if "test" in f and f.endswith(".parquet"))
    path = hf_hub_download(repo, file, repo_type="dataset", revision=revision)
    rows = pq.read_table(path).to_pylist()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prepared = []
    for i, row in enumerate(rows):
        messages = row["messages"]
        first_assistant = next((j for j, m in enumerate(messages) if m["role"] == "assistant"), len(messages))
        prompt = tokenizer.apply_chat_template(messages[:first_assistant], tokenize=True, add_generation_prompt=True)
        # Keep the beginning and the generation suffix when truncating long prompts.
        if len(prompt) > 512:
            prompt = prompt[:480] + prompt[-32:]
        prepared.append({"id": str(row.get("prompt_id", i)), "prompt": prompt, "max_new_tokens": 128, "arrival_ms": 0})
    rng = random.Random(42)
    corpus = [t for row in prepared for t in row["prompt"]]

    def fixed(size, i):
        start = (i * 2048) % (len(corpus) - size)
        return corpus[start:start + size]

    workloads = {"heldout": prepared,
        "b32-c2048": [{"id": str(i), "prompt": fixed(2048, i), "max_new_tokens": 128, "arrival_ms": 0} for i in range(32)],
        "mixed": [{"id": str(i), "prompt": fixed(rng.randint(128, 4096), i), "max_new_tokens": 64, "arrival_ms": i * 50} for i in range(64)],
        "lengths": [{"id": str(i), "prompt": fixed(rng.randint(512, 4096), i), "max_new_tokens": 16, "arrival_ms": 0} for i in range(128)]}
    workloads["spec-minimal"] = [{**row, "max_new_tokens": 64} for row in prepared[:16]]
    workloads["mixed-minimal"] = [{"id": str(i), "prompt": fixed(rng.randint(128, 1024), i),
        "max_new_tokens": 32, "arrival_ms": i * 25} for i in range(16)]
    hashes = {}
    for name, data in workloads.items():
        output = args.output / f"{name}.jsonl"
        output.write_text("\n".join(json.dumps(row) for row in data) + "\n", encoding="utf-8")
        hashes[name] = fingerprint(output)
    metadata = {"dataset": repo, "revision": revision, "split": "test", "file_sha256": fingerprint(path),
        "license": "CC-BY-NC-4.0", "source": "https://huggingface.co/datasets/HuggingFaceH4/no_robots",
        "prompts": len(prepared), "tokenizer": args.tokenizer, "seed": 42, "workload_sha256": hashes,
        "notes": ["Public evaluation prompts; absence from Qwen training data is not established.",
                  "heldout uses the first user turn/chat prefix, truncated to 512 tokens with generation suffix retained.",
                  "Fixed-length and mixed-load inputs concatenate tokenized prompts for controlled shapes; they are synthetic sequences of real text IDs."]}
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
