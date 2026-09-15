"""Render README figures from archived A100 measurement reports."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
INK = "#1e293b"
BLUE = "#4875cf"
DEEP_BLUE = "#294e96"
SLATE = "#64748b"
PALE = "#cbd5e1"
WASTE = "#e5bd83"
PAPER = "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
})


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def canvas(title, subtitle):
    fig, ax = plt.subplots(figsize=(12, 6.6), facecolor=PAPER)
    fig.subplots_adjust(left=.10, right=.96, bottom=.23, top=.75)
    fig.text(.06, .92, title, fontsize=22, weight="semibold")
    fig.text(.06, .855, subtitle, fontsize=11, color=SLATE)
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#edf0f4", linewidth=.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=10)
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", facecolor=fig.get_facecolor())
    fig.savefig(OUT / f"{name}.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def speculation():
    control = read("results/a100/prompt-lookup-control.json")["runs"][0]
    repetitive = read("results/a100/prompt-lookup-final.json")["runs"][0]
    values = [1, control["tpot_speedup"], repetitive["tpot_speedup"]]
    fig, ax = canvas("Repetition determines the payoff",
                      "Prompt lookup · TPOT speedup over each workload's target-only baseline · higher is better")
    bars = ax.bar(range(3), values, width=.55, color=[PALE, BLUE, DEEP_BLUE], edgecolor="none")
    ax.axhline(1, color=SLATE, linewidth=1, linestyle=(0, (5, 4)))
    ax.text(.985, 1.04, "1.0× baseline", transform=ax.get_yaxis_transform(),
            color=SLATE, ha="right", fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    ax.set_xticks(range(3), ["Target only\nNormalized baseline\nfor each workload",
                           "Lookup / general\n1 prompt · 64 tokens\n2 measured repeats",
                           "Synthetic repetitive\n8 prompts · 128 tokens each\n2 measured repeats"])
    ax.set_ylim(0, 3)
    ax.set_ylabel("TPOT speedup")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}×"))
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value+.08, f"{value:.2f}×",
                ha="center", fontsize=18, weight="bold")
    fig.text(.06, .075, "Qwen2.5-1.5B-Instruct · A100 40GB · greedy decoding · lookup lookahead 4", fontsize=10)
    fig.text(.06, .037, "Each speedup uses its own matched baseline. Synthetic and general prompts are different workloads.", fontsize=9)
    save(fig, "lookup-speedup")


def memory():
    report = read("results/a100/memory.json")
    rows = [report["runs"][0], report["runs"][2], report["runs"][-1]]
    assigned = [r["decode_snapshot"]["assigned_bytes"] / 2**30 for r in rows]
    live = [r["decode_snapshot"]["unique_live_bytes"] / 2**30 for r in rows]
    waste = [a-b for a, b in zip(assigned, live)]
    fig, ax = canvas("Allocate KV blocks as sequences grow",
                      "16 variable-length requests · assigned KV storage after decode · lower is better")
    ax.bar(range(3), live, width=.55, color=BLUE, edgecolor="none", label="Unique live KV payload")
    ax.bar(range(3), waste, bottom=live, width=.55, color=WASTE, edgecolor="none",
           label="Unused assigned capacity")
    ax.set_xticks(range(3), ["Contiguous\n4,096 slots / request", "Paged\n16-token blocks",
                           "Paged + sharing\n512-token shared prefix"])
    ax.set_ylim(0, 2.2)
    ax.set_ylabel("Assigned KV storage (GiB)")
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    for i, value in enumerate(assigned):
        ax.text(i, value+.065, f"{value:.3f} GiB", ha="center", fontsize=16, weight="bold")
    fig.text(.06, .095, "44.5% less assigned KV with paging; prefix sharing saves another 21.1%.", fontsize=12, weight="bold")
    fig.text(.06, .052, "Assigned blocks are not total GPU allocation: the paged implementation reserves a ~4 GiB backing pool.", fontsize=10)
    fig.text(.06, .021, "Source: results/a100/memory.json · Qwen2.5-1.5B FP16 · prompt lengths 699–4,041 + 4 decode tokens", fontsize=9)
    save(fig, "kv-memory")


if __name__ == "__main__":
    speculation()
    memory()
