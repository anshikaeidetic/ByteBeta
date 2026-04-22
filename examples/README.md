# Byte Examples

These examples use Byte.

## Safety note

Semantic caching can return a cached answer for a request that is similar but not actually equivalent. Treat semantic reuse as a guarded optimization, not a default assumption. For production-facing examples, prefer exact, normalized, or the stricter `init_safe_semantic_cache(...)` helper.

## Before you run

- install the repository from source and any backend-specific extras you need
- export the provider API keys required by the example you want to run
- keep the import path lowercase: `from byte import ...`
- expect some examples to require optional backends such as FAISS, SQLite, Redis, or provider SDKs

## Quick start

### Exact cache

```python
from byte import Cache, Config
from byte.adapter import ChatCompletion
from byte.adapter.api import init_exact_cache

cache_obj = Cache()
init_exact_cache(
    data_dir="example_exact_cache",
    cache_obj=cache_obj,
    config=Config(
        enable_token_counter=False,
        routing_model_aliases={"assistant": ["openai/gpt-4o-mini"]},
    ),
)

response = ChatCompletion.create(
    model="assistant",
    messages=[{"role": "user", "content": "Summarize Byte in one line."}],
    cache_obj=cache_obj,
)
print(response["choices"][0]["message"]["content"])
```

### Safety-first semantic cache

```python
from byte import Cache, Config
from byte.adapter.api import init_safe_semantic_cache

cache_obj = Cache()
init_safe_semantic_cache(
    cache_obj=cache_obj,
    data_dir="example_safe_semantic_cache",
    config=Config(enable_token_counter=False),
)
```

### Standalone gateway

```bash
byte_server --gateway True --gateway-mode adaptive --host 127.0.0.1 --port 8000 --cache-dir byte_data --security-mode --security-admin-token "<admin-token>"
```

Useful endpoints:

- `POST /put`
- `POST /get`
- `POST /flush`
- `GET /stats`
- `GET /metrics`
- `GET /healthz`
- `GET /readyz`
- `POST /v1/chat/completions`

## Example map

- [`adapter/api.py`](adapter/api.py): direct cache API helpers such as `put(...)`, `get(...)`, and the cache initializers
- [`context_process/selective_context.py`](context_process/selective_context.py): selective context assembly
- [`context_process/summarization_context.py`](context_process/summarization_context.py): context summarization flow
- [`processor/llm_verifier_example.py`](processor/llm_verifier_example.py): verification step before reuse
- [`processor/temperature_example.py`](processor/temperature_example.py): temperature-aware cache behavior
- [`session/session.py`](session/session.py): session isolation and scoped caching
- [`kubernetes/`](kubernetes): Kubernetes operator, CRD, and deployment examples
