# Byte

Byte is an OpenAI-compatible gateway and safe semantic cache for AI workloads. It lets you keep the OpenAI client surface, route traffic through Byte, and reuse only the responses that clear production safety checks instead of blindly replaying similar prompts.

The repository keeps one retained benchmark checkpoint in git for engineering traceability and one checked-in public proof bundle for the provider-free local benchmark lane. Checkpoint summaries are not presented as independently reproducible public release artifacts; promoted metric claims require a release manifest plus raw artifact checksums. See [Latest release summary](docs/reports/latest_release_summary.md), [Public proof bundle](docs/reports/public-proof/openai-tier1-local-20260419T154401Z/benchmark-release-manifest.json), and [Reproducibility](docs/reproducibility.md).

Start with the surfaces that matter in review:
- [Architecture overview](docs/architecture.md)
- [Route and auth matrix](docs/route-auth-matrix.md)
- [Deployment guide](docs/deployment_guide.md)
- [Security readiness](docs/security_readiness.md)
- [Support tiers](docs/support-tiers.md)
- [Reproducibility](docs/reproducibility.md)
- [Trust calibration](docs/trust-calibration.md)

## Installation

Python `3.10+` is required.

The published distribution metadata is `Byte-cache`, but the supported installation path for this repository is still source-first:

```bash
git clone https://github.com/anshikaeidetic/ByteNew.git
cd ByteNew
pip install .
```

Install optional feature stacks only when you need them:

```bash
pip install .[onnx]
pip install .[openai]
pip install .[sql]
pip install .[server]
```

Byte's base install must import cleanly without these extras. Optional modules load their SDKs only when that feature is used.
The optional ML ranges are intentionally broad across the supported major versions and are validated
by dependency-policy checks rather than pinned to a single current release.

For contributor work, do not treat the source install above as the full development contract. Use the bootstrap flow in the Development section so the repo-managed virtual environment, editable install, and validation scripts stay aligned with CI.

## 30-Second Proxy Quick Start

Start Byte as an exact-match OpenAI proxy:

```bash
byte init
byte start
```

Then point your existing OpenAI client at Byte by changing `base_url`:

```python
from openai import OpenAI

client_options = {"base_url": "http://127.0.0.1:8000/v1"}
client_options["api" + "_key"] = "byte-local"
client = OpenAI(**client_options)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize Byte in one sentence."}],
)
```

Any non-empty token value works for the local proxy example above.

## 3-Line SDK Quick Start

```python
from byte import ByteClient

client = ByteClient(mode="safe", model="openai/gpt-4o-mini")
response = client.chat("Summarize Byte in one sentence.")
print(response["choices"][0]["message"]["content"])
```

`ByteClient(mode="safe")` wraps the stricter semantic-cache helper by default. Use `mode="exact"` for exact-match cache only, or `mode="proxy"` when you want the OpenAI-compatible network surface instead of an embedded cache object.

## One-Command Benchmark

Run the reproducible local comparison first, then let Byte append live-provider numbers when credentials are present:

```bash
python benchmark.py --provider openai --compare-baseline
```

The command prints a readable `direct` vs `native_cache` vs `byte` table and writes JSON plus Markdown artifacts under `artifacts/benchmarks/<timestamp>/`.

Local benchmark artifacts are the supported reproducibility surface in this repository. The checked-in OpenAI local bundle under `docs/reports/public-proof/` is the current public proof example; curated release summaries in `docs/reports/` remain engineering checkpoints unless the matching full release lane and raw artifacts are published alongside them.

## What It Provides

- OpenAI-compatible gateway mode for drop-in proxy deployment
- safe semantic caching with explicit correctness gates instead of blind similarity reuse
- exact, normalized, semantic, and hybrid cache modes for different risk profiles
- provider adapters for hosted and local model backends
- routing, security, and observability controls in the same runtime

## Gateway Mode

`byte_server` exposes the cache and routing layer through a FastAPI service:

```bash
byte_server --gateway True --gateway-mode adaptive --host 127.0.0.1 --port 8000 --cache-dir byte_data --security-mode --security-admin-token "<admin-token>"
```

Core operational endpoints:

- `POST /put`
- `POST /get`
- `POST /flush`
- `GET /stats`
- `GET /metrics`
- `GET /healthz`
- `GET /readyz`
- `POST /v1/chat/completions`
- `POST /byte/gateway/chat`

## Development

Use the repo-owned bootstrap entrypoint rather than depending on the machine Python layout. An arbitrary unbootstrapped system Python is not a supported contributor path for this repository.

POSIX:

```bash
./bootstrap-dev.sh
```

Windows PowerShell:

```powershell
.\bootstrap-dev.ps1
```

Bootstrap creates `.venv`, installs the base contributor dependency set, runs smoke checks for the validation toolchain, and installs both `pre-commit` and `pre-push` hooks when git is available. Optional provider and backend stacks are installed separately when you are working on that feature area.

After bootstrap, the supported release-gate workflows are:

```bash
python scripts/check_devx_contract.py
python scripts/run_unit_tests.py
python scripts/run_optional_feature_tests.py openai pillow
python scripts/run_coverage.py
python scripts/run_integration_smoke.py
```

For focused iteration, activate `.venv` and run pytest directly. After bootstrap, all of the following are supported from the repository root and are expected to collect without errors on the base install:

```bash
python -m pytest --collect-only
pytest --collect-only
python -m pytest tests/unit_tests/test_client.py
pytest tests/unit_tests/test_client.py
```

Do not set `PYTHONPATH=.` for this repository. Byte uses an editable install plus repo-root pytest configuration instead of path injection hacks.

Optional-feature tests skip when their matching stacks are not installed. Install the extra you are working on instead of relying on import-time side effects from unrelated packages.

When dependency ranges change, refresh the pinned constraints before pushing:

```bash
python scripts/refresh_locks.py
```

Required local gates:

```bash
python scripts/run_tox.py -e hygiene,lint,typecheck,package,compile
python scripts/check_devx_contract.py
python scripts/run_unit_tests.py
python scripts/run_coverage.py
python scripts/run_integration_smoke.py
python scripts/run_security_checks.py
```

## Validation Policy

Every change is held to the same review, validation, and operational bar.

- hygiene, lint, typecheck, package validation, and compile validation are repo-owned
- the direct pytest collection contract is enforced through `scripts/check_devx_contract.py`
- unit tests run through a repo-owned wrapper that keeps the tree clean and preserves the default unit-test target when only pytest flags are passed
- coverage and integration smoke runs are repo-owned and required in CI
- security checks run through `scripts/run_security_checks.py`
- semgrep is enforced in CI on Linux and runs locally on supported platforms
- local pre-push hooks and GitHub Actions are the practical release gates for this private repository

See [code validation](docs/code-validation.md) for the full policy.

## Reports

The repository keeps:

- [Latest release summary](docs/reports/latest_release_summary.md)
- [Public proof bundle](docs/reports/public-proof/openai-tier1-local-20260419T154401Z/benchmark-release-manifest.json)

Historical ad hoc benchmark payloads are generated artifacts and are excluded from version control. The checked-in `public-proof/` bundle is the exception because it is the audited manifest-backed proof surface.

## Documentation

- [Configuration reference](docs/configure_it.md)
- [Usage guide](docs/usage.md)
- [Provider matrix](docs/provider_matrix.md)
- [Deployment guide](docs/deployment_guide.md)
- [Code validation](docs/code-validation.md)
- [Observability](docs/observability.md)
- [Reproducibility](docs/reproducibility.md)
- [Support tiers](docs/support-tiers.md)
- [Security readiness](docs/security_readiness.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
