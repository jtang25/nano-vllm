"""Parse Nsight Compute CSV. Never infer HBM bandwidth from tensor sizes alone."""
import argparse
import csv
import io
import json
from pathlib import Path

from benchmarks.experiment_utils import save


def counters(path):
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    start = next(i for i, line in enumerate(lines) if '"Metric Name"' in line or "Metric Name," in line)
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    total_bytes, kernel_seconds = 0., 0.
    seen = set()
    durations = {"nsecond": 1e-9, "ns": 1e-9, "usecond": 1e-6, "us": 1e-6, "msecond": 1e-3, "ms": 1e-3, "second": 1., "s": 1.}
    sizes = {"byte": 1., "bytes": 1., "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}
    for row in reader:
        name = row.get("Metric Name", "")
        if name not in ("dram__bytes.sum", "gpu__time_duration.sum"):
            continue
        key = (row.get("ID"), name)
        if key in seen:
            raise ValueError("duplicate kernel/metric rows; export one action range once")
        seen.add(key)
        value = float(row["Metric Value"].replace(",", ""))
        unit = row["Metric Unit"]
        if name == "dram__bytes.sum":
            total_bytes += value * sizes[unit]
        else:
            kernel_seconds += value * durations[unit]
    if not total_bytes or not kernel_seconds:
        raise ValueError("missing required DRAM bytes or kernel duration counters")
    return total_bytes, kernel_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncu-csv", required=True)
    parser.add_argument("--timing", required=True, help="matching unprofiled profile_workload JSON")
    parser.add_argument("--peak-tbps", type=float, required=True, help="actual GPU SKU peak, decimal TB/s")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    traffic, active = counters(args.ncu_csv)
    timing = json.loads(Path(args.timing).read_text())
    span = timing["decode_device_span_seconds"]
    if not span or args.peak_tbps <= 0:
        parser.error("requires CUDA device-span timing and positive peak bandwidth")
    report = {"measured_dram_bytes": traffic, "sum_profiled_kernel_seconds": active,
              "profiled_kernel_active_tbps": traffic / active / 1e12,
              "estimated_unprofiled_region_tbps": traffic / span / 1e12,
              "peak_tbps": args.peak_tbps, "region_fraction_of_peak": traffic / span / 1e12 / args.peak_tbps,
              "timing": timing,
              "caveats": ["Use counters filtered to exactly the same decode region/token count as timing.",
                          "Region bandwidth combines instrumented traffic with unprofiled time; replay/cache behavior can differ.",
                          "Kernel-active bandwidth excludes launch gaps and overlapping durations can double-count time.",
                          "Bandwidth utilization alone does not establish a memory-bound roofline; inspect compute utilization, launch gaps, and Nsight Compute roofline sections."]}
    save(args.output, report)


if __name__ == "__main__":
    main()
