# Reproducibility

Byte keeps a strict distinction between reproducible public surfaces, curated
engineering checkpoints, and internal research adaptation notes.

## What is reproducible in-repo

The following surfaces are expected to be runnable from this repository:

- `python benchmark.py --provider openai --compare-baseline`
- `byte benchmark`
- `build_workload_plan()`
- `provider_coverage()`

These are the supported public entrypoints for workload-shape inspection and
local comparison runs. They are versioned with the repository and covered by
regression tests.

## What is not a public proof artifact

`docs/reports/latest_release_summary.md` is a curated engineering checkpoint.
It is useful for tracking the most recent retained hardening lane, but it is
not a third-party reproducibility bundle and it is not a release-signoff
artifact on its own.

Byte now also checks in one public proof bundle for the provider-free local
comparison lane:

- `docs/reports/public-proof/openai-tier1-local-20260419T154401Z/`

That bundle includes a release manifest, a compact canonical raw-record JSON
artifact, summary artifacts, and checksums. It is the proof surface for the
current in-repo benchmark contract.

Numbers such as false reuse rate, confidence accuracy, or deterministic output
rate should be read in that context unless a matching reproducible release lane
and raw artifacts are published with the same commit.

## Public benchmark proof contract

A release-quality benchmark claim must include a `benchmark-release-manifest.json`
next to the published artifacts. The manifest must name the run ID, commit SHA,
profile, execution mode, providers, systems, raw record artifact, summary
artifact, checksums, environment metadata, generation time, release-gate result,
and public-proof status.

CI enforces this distinction with:

```bash
python scripts/check_benchmark_claims.py
```

Metric summaries without a manifest must use checkpoint language. The retained
DeepSeek numbers in `docs/reports/latest_release_summary.md` intentionally stay
in that category until raw records and a matching manifest are published. The
checked-in OpenAI local bundle satisfies the public-proof contract for that
specific lane only; it does not retroactively promote older checkpoint-only
summaries.

## Research references audited for this repository

Byte maps its registered research-backed features to primary sources and tracks
their implementation status separately from public benchmark claims.

- Prompt Cache: [arXiv:2311.04934](https://arxiv.org/abs/2311.04934)
- LongLLMLingua: [arXiv:2310.06839](https://arxiv.org/abs/2310.06839)
- LLMLingua-2: [arXiv:2403.12968](https://arxiv.org/abs/2403.12968)
- RECOMP: [arXiv:2310.04408](https://arxiv.org/abs/2310.04408)
- vCache: [arXiv:2502.03771](https://arxiv.org/abs/2502.03771)
- KV-cache compression survey: [MLSys 2025](https://proceedings.mlsys.org/paper_files/paper/2025/file/26289c647c6828e862e271ca3c490486-Paper-Conference.pdf)

The registry in `byte.research.registry` is the machine-readable source for the
current audit state. If a paper entry cannot be accurately matched to the
primary source, the registry should be corrected before it is treated as a
public truth surface.
