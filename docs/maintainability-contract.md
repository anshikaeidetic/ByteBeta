# Maintainability Contract

Byte treats type annotations, docstrings, and exception boundaries as release gates for the
maintained runtime surface. The contract is intentionally targeted: production and high-risk
validation modules are held to strict rules, while tests and legacy research implementations are
reported without forcing noisy annotation or docstring churn.

## Type Policy

Public helpers in maintained targets must be fully annotated. `self` and `cls` are the only
unannotated parameters allowed by the checker. The mypy strict surface remains the source of truth
for deeper type checking, and the maintainability gate keeps new high-risk modules from bypassing
that surface.

Every production function under `byte`, `byte_server`, `byte_inference`, and `byte_memory` must also
carry an explicit return annotation. The dedicated return-annotation contract is now blocking for
every tracked Python file in the repository, including repo-owned scripts, documentation helpers,
examples, and tests. Unknown dynamic provider boundaries may use a local `typing.Any` return, but
only where the underlying SDK object is intentionally dynamic.

## Documentation Policy

Every maintained target module must have a module docstring that explains responsibility. Exported
or nontrivial top-level public helpers must also have a concise behavior docstring. The gate does
not require docstrings for every private helper, trivial property, route closure, or test function;
those are reported in aggregate instead.

## Hotspot Decomposition Policy

Core runtime hotspots are tracked through `CORE_HOTSPOT_TARGETS` in
`byte._devtools.verification_targets`. These modules must stay split by responsibility rather than
growing back into broad implementation bundles. Compatibility facades stay below 250 lines, focused
provider/runtime orchestration modules stay below 450 lines, and pure helper modules stay below 400
lines unless an architecture test records a deliberate exception and rationale.

The current hotspot groups cover Gemini provider support, H2O runtime construction/generation,
model routing, optimization-memory summarization/stores, cache memory mixins, and response
finalization. Public imports remain stable; implementation modules own one reason to change.

## Exception Policy

Broad `except Exception` handlers are allowed only at explicit boundaries: provider SDK calls,
callback/observer execution, telemetry shutdown/export, best-effort cleanup, and HTTP route
normalization. Each allowed broad handler must be marked as a boundary catch and must preserve
structured context through logging, error classification, or a re-raise path.

## Required Command

Run the maintainability gate before pushing:

```bash
python scripts/check_maintainability_contract.py
python scripts/check_annotation_contract.py
```

The command prints the current AST metrics and fails when maintained targets lose module
documentation, public annotations, nontrivial helper docstrings, or exception-boundary
classification. CI runs the same command in the `maintainability` job.
