# Benchmark Architecture

Byte benchmark code is maintained as reproducible engineering infrastructure, not as ad hoc
research scripts. Public entrypoints remain stable while implementation detail is kept behind
private, provider-lazy modules.

## Boundaries

- Public command surfaces are `benchmark.py`, `byte benchmark`, and the script wrappers under
  `scripts/`. These entrypoints must stay thin and delegate to packaged benchmark modules.
- Public benchmark-plan surfaces are `build_workload_plan()` and `provider_coverage()` where they
  already exist. Plan modules must remain provider-free at import time.
- Private program implementations live under `byte.benchmarking._program_impl`. These modules own
  legacy orchestration logic and can be refactored internally without changing the public CLI.
- Workload synthesis is exposed through `byte.benchmarking.workload_generator` for compatibility,
  with provider-free family modules registered in `byte.benchmarking.workload_families`.
- Shared execution, request, and report primitives live in small typed modules so retry, accounting,
  and failure payload behavior is testable without live provider SDKs.

## Reproducibility Standard

Byte follows benchmark practices aligned with HELM and BetterBench: deterministic workload
construction, transparent scenario metadata, explicit provider coverage, and reproducible command
surfaces. Curated benchmark summaries are release checkpoints unless the full reproducible release
lane has run and its artifacts have been archived with the commit SHA, command line, dependency
profile, and generated reports.

## Optional Dependencies

Benchmark planning and workload construction must not import hosted-provider SDKs or live-service
clients. OpenAI, Anthropic, Groq, LangChain, Redis, and similar dependencies may be imported only
inside execution paths that are already running a provider-dependent benchmark lane. Base-install
collection must succeed with those modules absent or explicitly blocked.

## Maintenance Rules

- New benchmark entry modules are capped by the repository hygiene budget and should normally be
  below 400 lines.
- Shared scenario construction, execution, pricing, usage normalization, and report emission belong
  in internal benchmark modules, not in entrypoint scripts.
- Any future size exception requires a hygiene allowlist rationale and a unit test proving the rule.
- Provider-boundary failures should be recorded with provider, case id, phase, attempt count, error
  type, and message so benchmark reports are auditable rather than opaque.
