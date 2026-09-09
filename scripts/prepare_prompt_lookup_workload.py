"""Create a transparent repetitive-continuation workload for prompt lookup."""
import argparse
import json
from pathlib import Path

from model.checkpoint import load_checkpoint
from model.tokenizer import tokenizer_for
from benchmarks.workload import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()
    model = load_checkpoint(args.checkpoint)
    tokenizer = tokenizer_for(model).tokenizer
    patterns = [
        "red green blue yellow ",
        "alpha beta gamma delta epsilon ",
        "north east south west ",
        "one two three four five six ",
        "A1,B2,C3,D4,E5,",
        "01001101",
        "foo(bar); baz(qux); ",
        '{"status":"ok","value":1}\n',
    ]
    rows = []
    for index, pattern in enumerate(patterns):
        text = "Repeated continuation pattern:\n" + pattern * 32
        prompt = tokenizer(text, add_special_tokens=False).input_ids
        rows.append({"id": str(index), "prompt": prompt, "max_new_tokens": args.tokens, "arrival_ms": 0,
                     "pattern": pattern})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(args.output), "sha256": fingerprint(args.output), "requests": len(rows),
                      "kind": "synthetic repetitive sequences encoded with the pinned model tokenizer"}, indent=2))


if __name__ == "__main__":
    main()
