# Features

## Product Positioning

Byte is a semantic caching and gateway product. Customer-facing
materials should use the `Byte` product name consistently.

## Core Runtime

- Supports normal and streaming chat-completion caching
- Supports sync and async streaming across the surfaced hosted backends
- Supports top-k similar search results via the configured data manager
- Supports chained caches through `Cache.next_cache`
- Supports exact, normalized, semantic, and hybrid cache modes
- Supports Byte route-based backend routing, memory, and workflow layers
- Supports a local Hugging Face causal-generation backend with opt-in H2O
  KV-cache eviction for long-context text workloads

```python
bak_cache = Cache()
bak_cache.init()
cache.init(next_cache=bak_cache)
```

## Safety Controls

- Supports fully skipping a cache for a request with `cache_skip=True`
- Supports safe semantic guard rails through `init_safe_semantic_cache(...)`
- Uses token overlap, length ratio, canonical matching, and quality scoring to
  reduce false positives
- Uses stricter safe defaults for production semantic reuse

## Observability and Operations

- Supports point-in-time summaries through `/stats`
- Supports Prometheus-compatible metrics through `/metrics`
- Supports liveness and readiness probes through `/healthz` and `/readyz`
- Supports Byte-native gateway endpoints such as `/byte/gateway/chat`
- Supports OpenTelemetry OTLP export from `byte_server`
- Supports Datadog-friendly OTLP defaults for agent or collector based ingestion
- Supports Kubernetes operator driven deployments through `ByteCache` custom resources
- Supports secure gateway rate limits and bounded public/admin concurrency in secure mode

Current operational gaps that should be disclosed explicitly:

- No bundled dashboard or alerting UI
- No built-in autoscaling controller

H2O scope boundaries that should also be disclosed explicitly:

- Applies only to Byte-owned local Hugging Face autoregressive text generation
- Does not apply to hosted providers
- Does not apply to image, audio, speech, or moderation surfaces

## Composition Model

Users can assemble modules including:

- Byte Adapter: routes Byte model names and request payloads into configured backends
- Pre-processor: extracts stable cache keys and request context
- Context buffer: maintains session context
- Encoder: embeds text into dense vectors for similarity search
- Cache manager: searches, saves, and evicts cached data
- Ranker: evaluates similarity and cached-answer quality
- Post-processor: selects or shapes the returned answer
