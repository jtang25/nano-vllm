# A100 evidence

The small reports in this directory are suitable for source control. See
[`docs/results-a100.md`](../../docs/results-a100.md) for definitions, caveats,
commands, and resume wording.

The full raw copy is under `raw/qwen/` and includes the 85 MB Torch Profiler
trace, Nsight Systems report, logs, exploratory diagnostics, and final reports.
It is kept locally and ignored by Git.

`nano-vllm-a100-evidence-final.tar.gz` is the latest checkpoint-free snapshot
of the remote source, workloads, and raw evidence, including both proposer
strategies, the non-repetitive control, and their final measurements. Its
SHA-256 is:

`14f06d201d9cddbf88d577183b045bdab4ba69498a8e031fb5b583aa60549770`

The pre-control refactor archive is retained as
`nano-vllm-a100-evidence-refactor.tar.gz`. Its SHA-256 is:

`d4a45b333d62a3b417fed1d19860342f5bbc3781e6f99619f3fb082533efd120`

The earlier pre-refactor archive is retained as
`nano-vllm-a100-evidence.tar.gz`. Its SHA-256 is:

`3d7a53e555c712c1c8cd42692d1fa90a30a05802d80aaa9c8e56327feccec174`

The hash was computed remotely and matched after copying the archive to this
Windows workspace. Model checkpoints are reproducible from the pinned upstream
revisions and are excluded because of their size.
