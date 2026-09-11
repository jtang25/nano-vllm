import json
import os
import platform
from pathlib import Path
import statistics
import time
import torch

from benchmarks.workload import fingerprint


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(function, device):
    sync(device)
    start = time.perf_counter()
    result = function()
    sync(device)
    return result, start, time.perf_counter() - start


def environment(model, checkpoint, workload=None):
    root = Path(__file__).resolve().parents[1]
    sources = sorted(p for folder in ("model", "storage", "runtime", "scripts", "benchmarks")
                     for p in (root / folder).glob("*.py"))
    return {"checkpoint_sha256": fingerprint(checkpoint),
            "workload_sha256": fingerprint(workload) if workload else None,
            "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else "cpu",
            "dtype": str(model.dtype), "config": model.config.to_dict(),
            "attention_tile_tokens": int(os.environ.get("NANOVLLM_ATTENTION_TILE_TOKENS", "1024")),
            "runtime_attention_backends": [layer.attention.backend for layer in model.layers],
            "source_sha256": {p.relative_to(root).as_posix(): fingerprint(p) for p in sources},
            "parameters": sum(p.numel() for p in model.parameters())}


def latency(events, start):
    ttft = [(row[0] - start) * 1000 for row in events if row]
    tpot = [(row[-1] - row[0]) * 1000 / (len(row) - 1) for row in events if len(row) > 1]
    return {"ttft_ms": ttft, "tpot_ms": tpot, "median_ttft_ms": statistics.median(ttft) if ttft else None,
            "median_tpot_ms": statistics.median(tpot) if tpot else None}


def save(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {path}")
