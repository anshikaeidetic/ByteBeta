# Byte Environment Reference

Use environment variables for deployment-critical runtime settings. Use `byteai.toml` for local `byte start` bootstrap. Use CLI flags when you are managing a service entrypoint directly.

## Naming

Byte-owned deployment variables use the `BYTE_*` prefix. Provider SDKs keep their native variable names such as `OPENAI_API_KEY`.

## High-value runtime variables

### Gateway and control plane

- `BYTE_CONTROL_PLANE_DB`
- `BYTE_MEMORY_SERVICE_URL`
- `BYTE_BACKEND_API_KEY`

### Admin and security

- `BYTE_ADMIN_TOKEN`
- `BYTE_SECURITY_KEY`
- `BYTE_AUDIT_LOG_PATH`
- `BYTE_SECURITY_EXPORT_ROOT`
- `BYTE_REQUIRE_HTTPS`
- `BYTE_ALLOWED_EGRESS_HOSTS`

### Internal service auth

- `BYTE_INTERNAL_TOKEN`

### Inference worker

- `BYTE_WORKER_ID`
- `BYTE_WORKER_CACHE_DIR`
- `BYTE_WORKER_MODELS`
- `BYTE_WORKER_FREE_VRAM_GB`

### Observability

- `BYTE_OTEL_ENDPOINT`
- `BYTE_OTEL_HEADERS`
- `BYTE_OTEL_PROTOCOL`
- `BYTE_OTEL_ENVIRONMENT`
- `BYTE_OTEL_RESOURCE_ATTRIBUTES`
- `BYTE_DATADOG_AGENT_HOST`
- `BYTE_DATADOG_SERVICE`
- `BYTE_DATADOG_ENV`
- `BYTE_DATADOG_VERSION`

### Provider credentials

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `GEMINI_API_KEY`
- `GROQ_API_KEY`
- `DEEPSEEK_API_KEY`
- `MISTRAL_API_KEY`

## When to use what

- `byteai.toml`: local developer bootstrap through `byte init` and `byte start`
- env vars: deployment secrets and environment-specific wiring
- CLI flags: process-specific overrides and operator-rendered args

The repository root `.env.example` is the safe starting point for local and deployment scaffolding.
