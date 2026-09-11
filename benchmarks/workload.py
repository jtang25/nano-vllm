"""Save tokenized workloads once and reuse identical inputs across runtimes."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_workload(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not row["prompt"] or row["max_new_tokens"] < 1 for row in rows):
        raise ValueError("invalid workload")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--min-context", type=int)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--arrival-ms", type=float, default=0)
    parser.add_argument("--shared-prefix", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-lines", type=Path, help="one real held-out prompt per line; byte tokenize and truncate to context")
    args = parser.parse_args()
    minimum = args.min_context or args.context
    if min(args.requests, minimum, args.tokens, args.vocab) < 1 or not 0 <= args.shared_prefix <= minimum <= args.context:
        parser.error("invalid workload sizes")
    rng = random.Random(args.seed)
    prefix = [rng.randrange(args.vocab) for _ in range(args.shared_prefix)]
    rows = []
    lines = args.text_lines.read_text(encoding="utf-8").splitlines() if args.text_lines else None
    if lines is not None and len(lines) < args.requests:
        parser.error("text file has fewer lines than requests")
    for i in range(args.requests):
        size = rng.randint(minimum, args.context)
        prompt = prefix + [rng.randrange(args.vocab) for _ in range(size - len(prefix))]
        if lines is not None:
            from model.tokenizer import ByteTokenizer
            prompt = ByteTokenizer().encode(lines[i])[:args.context]
        rows.append({"id": str(i), "prompt": prompt,
                     "max_new_tokens": args.tokens, "arrival_ms": i * args.arrival_ms})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(args.output), "sha256": fingerprint(args.output), "synthetic": lines is None}))


if __name__ == "__main__":
    main()
