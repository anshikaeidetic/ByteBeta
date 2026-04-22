# Contributing

Byte is an infrastructure repository. Treat every change as production work.

## Development Workflow

1. Bootstrap the repo-owned environment with `./bootstrap-dev.sh` or `.\bootstrap-dev.ps1`.
2. Run repo-owned validation scripts from the repository root without setting `PYTHONPATH`.
   For focused test runs, activate `.venv` and use `python -m pytest` or `pytest` directly from the repository root after bootstrap. Both direct collection commands must succeed on the base install: `python -m pytest --collect-only` and `pytest --collect-only`.
   Use `python scripts/run_optional_feature_tests.py <feature>` when you need the repo-owned test slice for an optional stack such as `openai`, `groq`, `sqlalchemy`, or `transformers`.
3. Refresh the lockfiles when dependency ranges or extras change with `python scripts/refresh_locks.py`.
4. Make focused changes. Do not mix refactors, generated artifacts, and feature work in one commit.
5. Run the required local gates before pushing:
   - `python scripts/run_tox.py -e hygiene,lint,typecheck,maintainability,package,compile`
   - `python scripts/check_devx_contract.py`
   - `python scripts/run_unit_tests.py`
   - `python scripts/run_coverage.py`
   - `python scripts/run_integration_smoke.py`
   - `python scripts/run_security_checks.py`
   - `python scripts/check_maintainability_contract.py`
6. Push only after the local gates pass cleanly.
   On Windows, the security gate runs the local checks supported by the platform and CI enforces the Linux `semgrep` pass before merge.
   This repository is private and the current GitHub plan does not provide protected-branch rules here, so do not bypass the local hooks or the CI gates for direct pushes to `main`.

## Why No `PYTHONPATH`

This repository does not support `PYTHONPATH=.` as part of the normal development flow.
Bootstrap installs Byte in editable mode, and pytest resolves from the repository root through `pyproject.toml`.
That keeps local runs, tox, and CI on the same import path instead of depending on shell-specific path mutation.
The supported contributor contract is bootstrap-first rather than zero-bootstrap machine Python.

## Optional Features

Optional SDKs stay behind use-time loaders. Importing Byte should not require provider, ONNX, SQL, or storage extras that are unrelated to the current task.

- Install the extra that matches the feature you are touching, for example `pip install .[onnx]`.
- Do not add top-level imports that force optional packages to load during module import.
- Optional-feature tests should skip cleanly when the matching stack is absent, and feature lanes should be added in CI when the surface is public and supported.

## Change Standards

- Keep public behavior stable unless the change explicitly updates the contract.
- Prefer repo-owned scripts for release gates; use direct pytest only after bootstrap and without shell-specific path hacks.
- Keep lockfiles and constraints in sync with `pyproject.toml`.
- Remove dead code, unused imports, and placeholder text before review.
- Do not commit caches, virtual environments, build output, or benchmark raw payloads.
- Keep maintained runtime modules annotated, documented at module boundaries, and free of
  unclassified broad exception handlers.

## Benchmark Artifacts

The repository keeps one curated release summary under `docs/reports/latest_release_summary.md`.
Raw benchmark payloads, intermediate graphs, and historical report directories stay out of git.

## Pull Requests

- Keep titles short and factual.
- Describe the user-facing impact and the validation you ran.
- Call out compatibility risks, migrations, or security implications explicitly.
