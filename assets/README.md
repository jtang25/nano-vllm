# README visuals

`demo.gif` is the supplied A100 terminal recording. Its displayed values belong
to that coding demo, not the benchmark workloads plotted in the README.

`render_figures.py` builds SVG and PNG charts using Matplotlib. Run it from any
directory with `python assets/render_figures.py` from the repository root.
The figures read the committed measurement reports directly:

- `lookup-speedup`: `results/a100/prompt-lookup-final.json` and
  `results/a100/prompt-lookup-control.json`; each ratio uses its own workload's
  target-only baseline. The general-generation control is one prompt.
- `kv-memory`: `results/a100/memory.json`; decode snapshots show assigned bytes
  and unique live bytes. Unused assigned capacity is their difference. The
  backing pool allocation is reported separately and is not a plotted saving.

White backgrounds, blue bars, slate labels, and muted amber for unused capacity
keep the figures visually consistent. All figures use this project's data.
