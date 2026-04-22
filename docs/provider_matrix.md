# Byte Provider Matrix

This matrix covers the supported Byte surfaces and the feature boundaries that matter in production.

## Chat and completion backends

| Backend | Chat | Streaming | Routing | Memory integration | Local-runtime features |
| --- | --- | --- | --- | --- | --- |
| OpenAI | Yes | Yes | Yes | Yes | No |
| Anthropic | Yes | Yes | Yes | Yes | No |
| Gemini | Yes | Yes | Yes | Yes | No |
| Groq | Yes | Yes | Yes | Yes | No |
| OpenRouter | Yes | Yes | Yes | Yes | No |
| Ollama | Yes | Yes | Yes | Yes | Local daemon only |
| DeepSeek | Yes | Yes | Yes | Yes | No |
| Mistral | Yes | Yes | Yes | Yes | No |
| Cohere | Yes | Yes | Yes | Yes | No |
| Bedrock | Yes | Yes | Yes | Yes | No |
| Hugging Face | Yes | Yes | Yes | Yes | Yes |

## Public gateway surfaces

| Surface | Route family | Auth model |
| --- | --- | --- |
| Chat completions | `/v1/chat/completions` and `/byte/gateway/chat` | BYO provider token, or server-managed credentials behind the admin boundary |
| Images | `/byte/gateway/images` | same |
| Moderations | `/byte/gateway/moderations` | same |
| Speech | `/byte/gateway/audio/speech` | same |
| Audio transcription | `/byte/gateway/audio/transcriptions` | same |
| Audio translation | `/byte/gateway/audio/translations` | same |

## Byte-owned local runtime features

These features apply only when Byte controls the local runtime:

- worker affinity based on cached session state
- local Hugging Face execution through `byte_inference`
- H2O and Byte KV cache optimizations
- full reasoning-prefix reuse for local models

Hosted providers still benefit from:

- exact and semantic cache policy
- routing and trust layers
- memory summary reuse
- replay-backed recommendation and inspection

Byte does not claim control over provider-internal KV caches on hosted platforms.
