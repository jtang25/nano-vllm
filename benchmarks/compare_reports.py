import argparse
import json
from pathlib import Path


def compare(native, baseline):
    if native["identity"] != baseline["identity"] or native["gpu"] != baseline["gpu"]:
        raise ValueError("different weights/workload/settings/GPU: no valid throughput ratio")
    a, b = native["outputs"], baseline["outputs"]
    if not a or len(a) != len(b) or any(len(x) != len(y) or not x for x, y in zip(a, b)):
        raise ValueError("runtime output counts do not match")
    token_matches = sum(x == y for xs, ys in zip(a, b) for x, y in zip(xs, ys))
    return {"native_over_vllm_throughput": native["output_tokens_per_second"] / baseline["output_tokens_per_second"],
            "sequence_greedy_match": sum(x == y for x, y in zip(a, b)) / len(a),
            "token_greedy_match": token_matches / sum(map(len, a)),
            "note": "Investigate greedy divergence before presenting the ratio as an equivalent-output comparison."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--vllm", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(compare(json.loads(args.native.read_text()), json.loads(args.vllm.read_text())), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
