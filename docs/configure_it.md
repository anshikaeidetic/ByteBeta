# Configuration Reference

Byte keeps the public `Config` surface stable, but production code
should think in sections instead of a single flat field bag.

## Initialization Paths

Use one of these entrypoints:

1. `Cache.init(...)` for explicit wiring
2. `init_exact_cache(...)` for deterministic reuse
3. `init_normalized_cache(...)` for normalized exact reuse
4. `init_safe_semantic_cache(...)` for guarded semantic reuse
5. `init_hybrid_cache(...)` for exact -> normalized -> semantic chaining
6. `init_similar_cache_from_config(...)` for YAML-driven startup

## Sectioned Config Areas

| Section | Purpose |
| --- | --- |
| `observability` | OTLP, Datadog, and telemetry settings |
| `cache` | exact/semantic thresholds, TTL, tiering, and admission policy |
| `routing` | Byte route aliases, fallback order, retries, and adaptive routing |
| `quality` | verification, repair, and guarded reuse thresholds |
| `context_compiler` | auxiliary context budgeting, retrieval, and sketches |
| `prompt_distillation` | prompt compression, module reuse, and faithfulness verification |
| `memory` | execution memory, reasoning reuse, failure memory, and planner behavior |
| `budget` | latency/cost balancing targets |
| `security` | admin auth, HTTPS, export limits, redaction, egress, rate limits, concurrency |
| `compression` | vector and KV compression controls |
| `trust` | query risk, novelty handling, confidence calibration, and verifier policy |
| `integrations` | MCP timeouts and local-runtime integration flags |

## Security Fields That Matter in Production

These fields are now live and enforced by the gateway/runtime:

- `security_mode`
- `security_require_admin_auth`
- `security_admin_token`
- `security_require_https`
- `security_trust_proxy_headers`
- `security_allow_provider_host_override`
- `security_allowed_egress_hosts`
- `security_max_request_bytes`
- `security_max_upload_bytes`
- `security_max_archive_bytes`
- `security_max_archive_members`
- `security_rate_limit_public_per_minute`
- `security_rate_limit_admin_per_minute`
- `security_max_inflight_public`
- `security_max_inflight_admin`

## Prompt, Trust, and Compression

The current Byte production path also exposes:

- prompt distillation through `prompt_distillation_*`
- trust and deterministic execution through `trust_*`
- local runtime compression through `kv_codec`, `vector_codec`, and related fields

These settings are shared across benchmark, runtime, and server execution. They
are not benchmark-only flags.

## Recommended Production Semantic Defaults

```python
from byte import Config

config = Config(
    similarity_threshold=0.96,
    semantic_min_token_overlap=0.60,
    semantic_max_length_ratio=2.0,
    semantic_enforce_canonical_match=True,
    cache_admission_min_score=0.80,
    security_mode=True,
    security_require_https=True,
    security_rate_limit_public_per_minute=240,
    security_rate_limit_admin_per_minute=60,
    security_max_inflight_public=64,
    security_max_inflight_admin=16,
)
```

## YAML Configuration

Use `init_similar_cache_from_config(...)` when the runtime should be driven by
configuration files instead of Python code. Keep YAML limited to fields that are
actually consumed by the active runtime path, and validate the deployment with
the same tests or smoke checks you use for code-driven startup.
