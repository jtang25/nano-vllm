# Results

The curated A100 reports are in `a100/`; their interpretation and exact
reproduction commands are in `../docs/results-a100.md`.

cpu-smoke.json is a tiny functional run of the benchmark harness on CPU with
random weights, no warmup and one repetition. It is not performance evidence
and must not be used for resume numbers or backend rankings.

`a100/raw/` and the compressed evidence archive are intentionally ignored by
Git because the Torch trace is large. They remain in the local workspace. The
archive SHA-256 is recorded in `a100/README.md`. Curated JSON and logs retain
environment details, model settings, raw samples, source hashes, checkpoint
hashes, and workload hashes.
