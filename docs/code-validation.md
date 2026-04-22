# Code Validation

Byte applies the same approval standard to every change, regardless of how the draft originated.

## Approval Standard

Code can be drafted quickly. It is only accepted after review, validation, and runtime checks.

Before a change is pushed:

1. Verify the implementation manually.
2. Refresh lockfiles when dependency ranges or extras changed.
3. Run the repo-owned validation gates, including the direct pytest collection contract.
4. Confirm the change is minimal, readable, and consistent with the system boundary it touches.

## Required Local Gates

Run these commands from the repository root:

```bash
python scripts/run_tox.py -e hygiene,lint,typecheck,maintainability,package,compile
python scripts/check_devx_contract.py
python scripts/run_unit_tests.py
python scripts/run_optional_feature_tests.py openai pillow
python scripts/run_coverage.py
python scripts/run_integration_smoke.py
python scripts/run_security_checks.py
python scripts/check_maintainability_contract.py
python scripts/check_annotation_contract.py
python scripts/check_benchmark_claims.py
python scripts/check_trust_calibration.py
python scripts/check_dependency_policy.py
```

`scripts/check_devx_contract.py` is the developer-experience gate. In the bootstrapped repo-managed environment it must pass:

- `python -m pytest --collect-only`
- `pytest --collect-only`
- `python scripts/run_unit_tests.py --collect-only`

Optional feature stacks are validated in dedicated repo-owned lanes rather than by widening the base unit environment. Use `scripts/run_optional_feature_tests.py` for that slice locally and keep the base install lean.

`scripts/check_maintainability_contract.py` is the type, documentation, and exception-boundary
gate for maintained runtime surfaces. It reports repo-wide AST metrics, enforces module docstrings
and public annotations on the maintained target set, and rejects broad exception handlers unless
they are explicit boundary catches with structured context.

`scripts/check_annotation_contract.py` is stricter than the aggregate maintainability ratio: every
tracked Python function in git must have an explicit return annotation, including production
packages, repo-owned scripts, documentation helpers, examples, and tests.

`scripts/check_benchmark_claims.py` enforces truthful benchmark wording. Public metric claims must
either be backed by a release manifest with raw artifacts and checksums or be clearly labeled as
engineering checkpoints.

`scripts/check_trust_calibration.py` validates the trust calibration artifact checksum, rejects
private validation metrics in internal checkpoint artifacts, requires a manifest for public-proof
calibration status, and rejects hidden trust-score floats in scoring modules. Score changes belong
in the versioned artifact.

`scripts/check_dependency_policy.py` verifies the optional ML dependency ranges and their declared
minimum/latest compatibility smoke versions.

Core hotspot refactors are also covered by the architecture-hardening tests and the lint/typecheck
target lists. When a hotspot is split, add each facade and extracted module to
`CORE_HOTSPOT_TARGETS`, keep line budgets ratcheted in `test_architecture_hardening.py`, and verify
that compatibility imports still resolve without hosted-provider optional dependencies.

## What the Security Gate Covers

`scripts/run_security_checks.py` runs:

- `bandit` on maintained production Python surfaces
- `pip-audit` on the installed dependency set
- `semgrep` with security-focused rules on maintained production Python surfaces when the platform supports a local Semgrep install, with CI enforcing the same scan on Linux
- `detect-secrets` against the repository content that should never contain credentials

## Review Expectations

- Validate failure paths, not only happy paths.
- Confirm new dependencies are justified and captured in the lockfiles.
- Remove dead code, unused helpers, and redundant abstractions.
- Reject changes that pass mechanically but are still unclear, over-broad, or unsafe.
