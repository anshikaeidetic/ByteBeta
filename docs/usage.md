# Byte Usage

Byte is easiest to adopt in one of three ways:

1. `ByteClient(mode="safe")` for embedded guarded semantic reuse.
2. `byte start` or `byte_server` for an OpenAI-compatible proxy.
3. A split-service deployment when you want the control plane, private inference workers, and the memory service separated.

## Default Library Path: `ByteClient(mode="safe")`

```python
from byte import ByteClient

client = ByteClient(mode="safe", model="openai/gpt-4o-mini")
response = client.chat("Summarize Byte in one sentence.")
content = response["choices"][0]["message"]["content"]
```

`ByteClient(mode="safe")` owns a local cache initialized through `init_safe_semantic_cache(...)`.
`ByteClient.chat()` returns the same OpenAI-compatible JSON payload in `safe`, `exact`, and `proxy` modes.

Choose a mode based on operational risk:

- `safe`: guarded semantic reuse with stricter correctness gates
- `exact`: exact-match cache only
- `proxy`: no embedded cache object, just the public `/v1/chat/completions` surface

The legacy `byte.client.Client` API remains supported for existing callers. New code should prefer `ByteClient`. Async callers should use `Client.aput(...)`, `Client.aget(...)`, and `Client.achat(...)`.

## Drop-In Proxy Path

Start the gateway:

```bash
byte init
byte start
```

Then keep the OpenAI SDK surface and only change `base_url`:

```python
from openai import OpenAI

client_options = {"base_url": "http://127.0.0.1:8000/v1"}
client_options["api" + "_key"] = "byte-local"
client = OpenAI(**client_options)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is Byte?"}],
)
```

Any non-empty token value works for the local proxy example above.

Public compatibility routes:

- `POST /v1/chat/completions`
- `POST /byte/gateway/chat`

## Split-Service Runtime

Use the split-service topology when you want a public control plane with private workers and a private memory service.

Start the memory service:

```bash
byte_memory --host 127.0.0.1 --port 8091 --internal-auth-token "<internal-token>"
```

Start an inference worker:

```bash
byte_inference \
  --host 127.0.0.1 \
  --port 8090 \
  --worker-id worker-a \
  --models huggingface/* \
  --internal-auth-token "<internal-token>"
```

Start the public control plane:

```bash
byte_server \
  --gateway True \
  --host 127.0.0.1 \
  --port 8000 \
  --cache-dir byte_data \
  --memory-service-url http://127.0.0.1:8091 \
  --control-plane-worker-url http://127.0.0.1:8090 \
  --internal-auth-token "<internal-token>"
```

See:

- [Service topology](service-topology.md)
- [Control plane](control-plane.md)
- [Inference worker](inference-worker.md)
- [Memory service](memory-service.md)

## Benchmark the Product Surface

```bash
python benchmark.py --provider openai --compare-baseline
```

That command always runs a reproducible local comparison first, then appends a live-provider section when credentials are available. It prints a readable table and writes JSON plus Markdown artifacts to `artifacts/benchmarks/<timestamp>/`.

## Advanced: low-level init helpers

If you need to wire cache objects directly, the low-level helpers still exist and remain supported.

### Exact match cache

```python
from byte import Cache, Config
from byte.adapter import ChatCompletion
from byte.adapter.api import init_exact_cache

cache_obj = Cache()
init_exact_cache(
    data_dir="byte_cache",
    cache_obj=cache_obj,
    config=Config(
        enable_token_counter=False,
        routing_model_aliases={"assistant": ["openai/gpt-4o-mini"]},
    ),
)

response = ChatCompletion.create(
    model="assistant",
    messages=[{"role": "user", "content": "what is github"}],
    cache_obj=cache_obj,
)
```

### Build your own cache pipeline

```python
from byte import cache
from byte.embedding import Onnx
from byte.manager import CacheBase, VectorBase, get_data_manager
from byte.similarity_evaluation.distance import SearchDistanceEvaluation

onnx = Onnx()
data_manager = get_data_manager(
    CacheBase("sqlite"),
    VectorBase("faiss", dimension=onnx.dimension),
)

cache.init(
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
)
```

## Related Docs

- [Architecture overview](architecture.md)
- [Deployment guide](deployment_guide.md)
- [Environment reference](environment-reference.md)
- [Route and auth matrix](route-auth-matrix.md)
