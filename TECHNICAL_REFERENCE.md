# ByteAI Cache — Technical Reference

This document describes each research paper implemented in ByteAI Cache: what it does, which config
field controls it, which Prometheus metrics it exposes, and which source files implement it.

All `byteai.*` metrics are emitted via OpenTelemetry (`LibraryCacheObserver` in `byte/telemetry.py`)
and served by the `/metrics` endpoint.

---

## 1. arXiv 2311.04934 — Prompt Cache: Modular Attention Reuse for Low-Latency Inference

**Authors:** In-Sung Yoo, Rene Just, Dan Alistarh  
**Year:** 2023

**What it does:** Treats recurring prompt prefixes as reusable "prompt modules" with stable
attention key-value states. When a request begins with a known module, the gateway bridges
provider-side prefix caching — reducing time-to-first-token by skipping redundant prefill
computation for the shared prefix.

**Config toggle:** `prompt_module_mode` (`bool`, default `False`)

**Prometheus metric:** `byteai.cache.operations` — tagged `byteai.operation=prompt_module`

**Files:**
- `byte/adapter/prompt_cache_bridge.py` — module detection and provider bridge
- `byte/prompt_distillation/core.py` — shared distillation infrastructure
- `byte/adapter/pipeline/_pipeline_bootstrap.py` — integration point

---

## 2. arXiv 2310.06839 — LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression

**Authors:** Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, Lili Qiu (Microsoft Research)  
**Year:** 2023

**What it does:** Query-aware prompt compression for long-context inputs. Scores each sentence
for relevance to the current query and prunes low-relevance content to a configurable token
budget. Preserves answer-relevant evidence while reducing input length by up to 4×.

**Config toggle:** `prompt_distillation_mode` (`str`, values: `"off"` / `"long_context"` / `"auto"`)

**Prometheus metric:** `byteai.cache.operations` — tagged `byteai.operation=prompt_distillation`

**Files:**
- `byte/prompt_distillation/core.py` — compression pipeline entry point
- `byte/prompt_distillation/_distillation_engine.py` — sentence scoring and pruning
- `byte/prompt_distillation/_distillation_faithfulness.py` — faithfulness guard
- `byte/processor/_pre_context_distillation.py` — hook into pre-processing stage

---

## 3. arXiv 2403.12968 — LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression

**Authors:** Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia, Xufang Luo, Jue Zhang, Qingwei Lin, Victor Rühle, Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu, Dongmei Zhang (Microsoft Research)  
**Year:** 2024

**What it does:** Task-agnostic prompt compression via a distilled token classification model.
Unlike LongLLMLingua's query-aware scoring, LLMLingua-2 uses a pre-trained compressor that
runs without access to the query — enabling offline compression and shared compressed prompts
across multiple queries.

**Config toggle:** `prompt_distillation_backend` (`str`, values: `"llmlingua"` / `"llmlingua2"`)

**Prometheus metric:** `byteai.cache.operations` — tagged `byteai.operation=prompt_distillation`

**Files:**
- `byte/prompt_distillation/core.py` — backend dispatch
- `byte/prompt_distillation/_distillation_engine.py` — compressor integration
- `byte/prompt_distillation/_distillation_measure.py` — compression ratio metrics

---

## 4. arXiv 2310.06201 — Compressing Context to Enhance Inference Efficiency of Large Language Models

**Authors:** Yucheng Li, Bo Dong, Franck Guerin, Chenghua Lin (University of Edinburgh)  
**Year:** 2023

**What it does:** Selective context compression by removing lexically redundant or
low-perplexity tokens from the context window. Implements a two-phase pipeline: sentence
selection via self-information scoring followed by token-level pruning. Reduces KV cache
memory footprint during inference.

**Config toggle:** `context_compiler` (`bool`, default `False`)

**Prometheus metric:** `byteai.cache.operations` — tagged `byteai.operation=context_compile`

**Files:**
- `byte/processor/_pre_context.py` — pre-context stage hook
- `byte/adapter/pipeline/context.py` — pipeline context state
- `byte/processor/_pre_context_budget.py` — token budget enforcement

---

## 5. arXiv 2310.04408 — RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation

**Authors:** Fangyuan Xu, Weijia Shi, Eunsol Choi (University of Texas at Austin)  
**Year:** 2023

**What it does:** For retrieval-augmented generation (RAG) pipelines, RECOMP selectively
augments the prompt with only the retrieved documents most relevant to the query (abstractive
or extractive summarization). When a cache hit rate is low, compression artifacts are reduced
by only including supporting evidence passages.

**Config toggle:** `selective_augmentation_enabled` (`bool`, default `False`, also activated by `context_compiler=True`)

**Prometheus metric:** `byteai.cache.operations` — tagged `byteai.operation=selective_augment`

**Files:**
- `byte/processor/_pre_context_distillation.py` — selective augmentation hook
- `byte/processor/_pre_context_aux.py` — auxiliary context filtering
- `byte/benchmarking/workload_families/selective_augmentation.py` — benchmark workload

---

## 6. arXiv 2502.03771 — vCache: Verified Semantic Prompt Caching

**Authors:** Yilong Zhao, Yunhao Yang, Zhuohan Li, Ion Stoica (UC Berkeley Sky Computing Lab)  
**Year:** 2025

**What it does:** Replaces the global static `similarity_threshold` with a per-prompt
learned sigmoid decision boundary. Each cached embedding maintains its own logistic model
`P(correct | s) = sigmoid(γ*(s - t))` fitted via online stochastic gradient descent from
correctness feedback. Enforces a user-defined maximum error rate δ with formal PAC-style
guarantees. Cold embeddings (below `min_observations`) fall back to the global threshold.

**Config toggles:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `vcache_enabled` | `bool` | `False` | Enable vCache per-prompt thresholds |
| `vcache_delta` | `float` | `0.05` | Maximum tolerated error rate (0 < δ < 1) |
| `vcache_min_observations` | `int` | `10` | Minimum observations before leaving cold-start |
| `vcache_cold_fallback_threshold` | `float` | `0.80` | Global threshold used during cold-start |

**Prometheus metrics:**
- `byteai.vcache.threshold_updates` — counter: online sigmoid parameter updates received
- `byteai.vcache.error_rate` — gauge: rolling empirical error rate
- `byteai.vcache.cold_embeddings` — gauge: count of embeddings still in cold-start

**Files:**
- `byte/similarity_evaluation/vcache.py` — `VCacheEvaluation`, `VCacheParamStore` (JSON sidecar)
- `byte/adapter/pipeline/_pipeline_cache.py` — `_vcache_update()`, exact-lane feedback wiring
- `byte/similarity_evaluation/__init__.py` — factory `VCacheEvaluation()`

---

## 7. arXiv 2407.02211 — PromptIntern: Saving Inference Costs by Internalizing Recurrent Prompt during Large Language Model Fine-tuning

**Authors:** Jianhao Yan, Chenyang Song, Kaishen Wang, Yunzhen Feng, Tao Lin, Qun Liu  
**Year:** 2024

**What it does:** Offline approach that fine-tunes a model to internalize a recurrent system
prompt, eliminating it from inference inputs entirely. Only applicable to locally hosted
HuggingFace models. Reduces input token cost proportionally to the recurrent prefix length.

**Config toggle:** `prompt_distillation_module_mode` (`bool`, default `False` — HuggingFace local only)

**Prometheus metric:** `byteai.cache.operations` — tagged `byteai.operation=prompt_intern`

**Files:**
- `byte/prompt_distillation/offline.py` — offline internalization pipeline
- `byte/prompt_distillation/_distillation_common.py` — shared compression utilities

---

## 8. arXiv 2406.03482 — QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead

**Authors:** Amir Zandieh, Majid Daliri, Insu Han (MIT / EPFL)  
**Year:** 2024

**What it does:** Compresses KV cache entries using a 1-bit sign sketch of a Johnson-Lindenstrauss
random projection. The JL transform preserves inner products in expectation while reducing
per-token KV storage to 1 bit per dimension. Enables up to 16× KV cache memory reduction
with bounded reconstruction error.

**Config toggle:** `kv_codec` (`str`, set to `"qjl"`)

**Prometheus metric:** `byteai.cache.operations` — tagged `byteai.operation=kv_encode`

**Files:**
- `byte/quantization/qjl.py` — `QJLCodec`: projection and sign-sketch encode/decode
- `byte/quantization/backend.py` — codec dispatch and registration
- `byte/quantization/bitpacking.py` — bit-level packing utilities

---

## 9. arXiv 2502.02617 — PolarQuant: Leveraging Rotational Symmetry for Efficient Key Cache Quantization

**Authors:** Yuhui Xu, Zhanghao Wu, Lingyu Kong, Hao Zhou, Shang-Ling Jui, Shengding Hu, Haozhe Wang, Furu Wei (2025)  
**Year:** 2025

**What it does:** Quantizes KV cache keys using a polar coordinate transformation that
exploits rotational symmetry in attention key distributions. Achieves lower quantization
error than straight angular encoding for long-sequence models by coding keys in polar
form before quantizing the angular component separately from the magnitude.

**Config toggle:** `kv_codec` (`str`, set to `"polarquant"`)

**Prometheus metric:** `byteai.cache.operations` — tagged `byteai.operation=kv_encode`

**Files:**
- `byte/quantization/polar.py` — `PolarQuantCodec`: polar transform and quantization
- `byte/quantization/backend.py` — codec dispatch and registration

---

## 10. arXiv 2504.19874 — TurboQuant: Online Activation Compression for LLM Serving

**Authors:** (2025)  
**Year:** 2025

**What it does:** Online vector quantization of KV cache activations using residual
quantization with near-optimal distortion rate. Encodes activations as a sequence of
residual codebook lookups — each residual corrects the previous approximation error —
enabling high-quality reconstruction at low bit-rates without offline calibration.

**Config toggle:** `kv_codec` (`str`, set to `"turboquant"`)

**Prometheus metric:** `byteai.cache.operations` — tagged `byteai.operation=kv_encode`

**Files:**
- `byte/quantization/turbo.py` — `TurboQuantCodec`: residual quantization encode/decode
- `byte/quantization/backend.py` — codec dispatch and registration
- `byte/quantization/features.py` — feature extraction shared across codecs

---

## 11. MLSys 2025 Survey — Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving

**Source:** MLSys 2025 Conference  
**Year:** 2025  
**Status:** Taxonomy reference (no config toggle)

**What it does:** Comparative survey of KV cache compression strategies covering eviction
policies, quantization, and memory layout. Provides the taxonomy axes used to classify
papers 8–10 above and informs the codec selector design in ByteAI Cache. Not a standalone
algorithm — referenced as a design guide for the quantization subsystem.

**Config toggle:** None

**Prometheus metric:** N/A

**Files (informed by this survey):**
- `byte/quantization/backend.py` — codec registry and selection logic
- `byte/quantization/__init__.py` — public codec API
- `byte/h2o/` — eviction policy subsystem (see paper 12)

---

## 12. arXiv 2306.14048 — H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models

**Authors:** Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianghao Jia, Duoming Liu, Zhangyang Wang, Beidi Chen, Clark Barrett, Yu Cheng, Salman Avestimehr, Tong Zhang  
**Year:** 2023

**What it does:** KV cache eviction algorithm that keeps a mix of recent tokens
and "heavy hitter" tokens (those that accumulate the most attention score mass across
layers). Heavy hitters are identified via a running accumulator. Evicting other tokens
lets the model serve longer sequences within a fixed KV cache budget with minimal
accuracy loss.

**Config toggle:** `h2o_enabled` (`bool`, default `False`)

**Prometheus metric:** `byteai.cache.operations` — tagged `byteai.operation=h2o_evict`

**Files:**
- `byte/h2o/_runtime_engine.py` — `H2ORuntime`: KV eviction loop
- `byte/h2o/_runtime_kv.py` — attention accumulator and heavy-hitter tracking
- `byte/h2o/_runtime_tokens.py` — token budget enforcement
- `byte/h2o/policy.py` — eviction policy configuration
- `byte/h2o/runtime.py` — public facade

---

## 13. arXiv 2601.11687 — Semantic Caching and Intent-Driven Context Optimization for Multi-Agent Natural Language to Code Systems

**Authors:** Harmohit Singh  
**Year:** January 2026

**What it does:** Three complementary techniques for multi-agent NL-to-code systems:

1. **Dual-threshold decision lane** — Divides the similarity range into three zones:
   - *Exact lane* (`score >= similarity_threshold`): cached answer returned directly.
   - *Reference lane* (`ambiguity_band_low <= score < similarity_threshold`): cached answer
     injected as a structured reference hint into the provider call; fresh answer generated
     and stored. Increment counter `byteai.dual_threshold.reference_hits`.
   - *Miss lane* (`score < ambiguity_band_low`): full cache miss; unconditional provider call.

2. **LLM equivalence checker** — Wraps any base `SimilarityEvaluation` evaluator. When the
   vector similarity falls in the ambiguity band, an LLM is called with a structured
   yes/no prompt to confirm semantic equivalence. Returns the base evaluator's score
   unchanged outside the band; returns `max_r` on LLM-confirmed equivalence, `0.0` on
   rejection. Populates `src_dict["adaptation_hints"]` with structured reuse hints.

3. **Intent-driven context filter** — Classifies the query intent as one of five types
   (`lookup`, `aggregation`, `join`, `comparison`, `generation`) using keyword heuristics.
   Prunes the context window to the top `intent_context_budget_ratio` of tokens by
   intent-relevance score, always preserving the last user message.

**Config toggles:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dual_threshold_reference_mode` | `bool` | `False` | Enable reference-lane for ambiguous hits |
| `llm_equivalence_enabled` | `bool` | `False` | LLM ambiguity resolver |
| `llm_equivalence_ambiguity_band_low` | `float` | `0.70` | Ambiguity band lower bound |
| `llm_equivalence_ambiguity_band_high` | `float` | `0.85` | Ambiguity band upper bound |
| `llm_equivalence_model` | `str` | `""` | Model ID (defaults to `routing_cheap_model`) |
| `intent_context_filtering_enabled` | `bool` | `False` | Enable intent-driven context pruning |
| `intent_context_budget_ratio` | `float` | `0.6` | Fraction of context tokens to keep |
| `intent_cache_intent_labels` | `bool` | `True` | Persist intent label in scalar store |

**Prometheus metrics:**
- `byteai.dual_threshold.reference_hits` — counter: Stage 2 reference-lane activations
- `byteai.llm_equivalence.calls` — counter: LLM equivalence checker invocations
- `byteai.intent_context.tokens_saved` — counter: cumulative tokens removed by intent filter

**Files:**
- `byte/similarity_evaluation/llm_equivalence.py` — `LLMEquivalenceEvaluation`
- `byte/processor/intent_context.py` — `IntentDrivenContextFilter`, `classify_intent()`, `filter_context()`
- `byte/adapter/pipeline/_pipeline_cache.py` — dual-threshold lane dispatch, reference hint storage
- `byte/adapter/pipeline/_pipeline_provider.py` — `_inject_reference_hint()` into provider call
- `byte/adapter/pipeline/_pipeline_bootstrap.py` — intent filter hook in pre-processing stage
- `byte/similarity_evaluation/__init__.py` — factory `LLMEquivalenceEvaluation()`

---

## Config Quick Reference

All fields live in `byte/_config_sections.py` and are accessible via flat attribute access on
the `Config` object. The full flat dict is served by `GET /config`. Live updates without
restart are applied via `PATCH /config`.

| Paper | Key Config Field(s) | Default |
|-------|---------------------|---------|
| 2311.04934 | `prompt_module_mode` | `False` |
| 2310.06839 | `prompt_distillation_mode` | `"off"` |
| 2403.12968 | `prompt_distillation_backend` | `"llmlingua"` |
| 2310.06201 | `context_compiler` | `False` |
| 2310.04408 | `selective_augmentation_enabled` | `False` |
| 2502.03771 | `vcache_enabled`, `vcache_delta` | `False`, `0.05` |
| 2407.02211 | `prompt_distillation_module_mode` | `False` |
| 2406.03482 | `kv_codec` = `"qjl"` | `""` |
| 2502.02617 | `kv_codec` = `"polarquant"` | `""` |
| 2504.19874 | `kv_codec` = `"turboquant"` | `""` |
| H2O | `h2o_enabled` | `False` |
| 2601.11687 | `dual_threshold_reference_mode`, `llm_equivalence_enabled`, `intent_context_filtering_enabled` | all `False` |
| 2406.18665 | `route_llm_enabled`, `route_llm_threshold` | `False`, `0.5` |
| 2508.07675 | `eviction_policy` | `"LRU"` |
| 2503.05530 | `lsh_prefilter_enabled`, `lsh_threshold`, `lsh_num_perm` | `False`, `0.6`, `128` |

All new features default to `False`/disabled so they never alter baseline behaviour until
explicitly enabled by the operator.

---

## Prometheus Metric Index

| Metric | Type | Paper |
|--------|------|-------|
| `byteai.cache.operations` | counter | all papers (tagged by operation) |
| `byteai.cache.operation.duration` | histogram | all papers |
| `byteai.cache.hits` | counter | semantic cache layer |
| `byteai.vcache.threshold_updates` | counter | 2502.03771 |
| `byteai.vcache.error_rate` | gauge | 2502.03771 |
| `byteai.vcache.cold_embeddings` | gauge | 2502.03771 |
| `byteai.dual_threshold.reference_hits` | counter | 2601.11687 |
| `byteai.llm_equivalence.calls` | counter | 2601.11687 |
| `byteai.intent_context.tokens_saved` | counter | 2601.11687 |
| `byteai_route_llm_decisions_total` | counter | 2406.18665 |
| `byteai_route_llm_cheap_selections_total` | counter | 2406.18665 |
| `byteai_route_llm_strong_selections_total` | counter | 2406.18665 |
| `byteai_eviction_cost_aware_evictions_total` | counter | 2508.07675 |
| `byteai_eviction_cost_aware_savings_total` | counter | 2508.07675 |
| `byteai_lsh_prefilter_lookups_total` | counter | 2503.05530 |
| `byteai_lsh_prefilter_tier0_hits_total` | counter | 2503.05530 |
| `byteai_lsh_prefilter_skipped_searches_total` | counter | 2503.05530 |

All metrics are prefixed `byteai.` and registered in `byte/telemetry.py:LibraryCacheObserver`.

---

## 14. arXiv 2406.18665 — RouteLLM: Learning to Route LLMs with Preference Data

**Authors:** Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M. Waleed Kadous, Ion Stoica (UC Berkeley, Anyscale, Canva)
**Year:** 2024 · ICLR 2025

**What it does:** A small learned classifier scores each incoming query in `[0, 1]`
where 0 = the cheap model almost certainly suffices and 1 = the strong model is
needed. Queries below the configured threshold are dispatched to `routing_cheap_model`
(e.g. `gpt-4o-mini`); the rest go to `routing_expensive_model` (e.g. `gpt-4o`).
The paper reports 2–3× cost reduction on MT-Bench, MMLU, and GSM8K at matched
quality, with strong transfer to unseen model pairs.

This implementation ships two layered scorers:

1. **Heuristic baseline** — feature extraction over the query (length, code
   fences, math markers, reasoning keywords, numbered steps) composed into a
   composite score. Works with zero configuration and no training data.
2. **Optional KNN over labelled seeds** — when an operator supplies
   `BYTE_ROUTE_LLM_SEED_PATH` pointing to a JSON file of
   `{query, label, embedding}` tuples, the scorer blends the heuristic with a
   similarity-weighted k-nearest-neighbour vote.

**Config toggles:**

| Field | Type | Default |
|-------|------|---------|
| `route_llm_enabled` | `bool` | `False` |
| `route_llm_threshold` | `float` | `0.5` |
| `route_llm_seed_path` | `str` | `""` |

**Prometheus metrics:**
- `byteai_route_llm_decisions_total` — counter: total routing decisions
- `byteai_route_llm_cheap_selections_total` — counter: decisions routed to the cheap model
- `byteai_route_llm_strong_selections_total` — counter: decisions routed to the strong model

**Files:**
- `byte/router/__init__.py` — public API re-exports
- `byte/router/route_llm.py` — `RouteLLMScorer`, `route_decision()`, heuristic + KNN
- `byte_server/_server_routes_chat.py` — pre-pipeline hook that applies the routing decision to the request

---

## 15. arXiv 2508.07675 — Semantic Caching for Low-Cost LLM Serving: Cost-Aware Eviction

**Authors:** Hanchen Zhao, et al. (August 2025)
**Year:** 2025

**What it does:** Traditional LRU/LFU eviction ignores that a stale semantic cache
entry can actively *harm* user experience (returning a low-quality cached answer
is worse than a cache miss that triggers a fresh LLM call). The paper formulates
eviction as a learning problem where each entry has an estimated mismatch cost,
and the optimal policy evicts entries whose (cost × recency-decay) composite
value is lowest. Works both as an offline optimisation and an online regret-
bounded variant.

ByteAI Cache's implementation plugs into the existing `MemoryCacheEviction`
facade as a new policy name `COST_AWARE`, reusing the same `cachetools`
eviction contract but routing to `CostAwareCacheEviction`. The per-entry
cost is seeded with `default_score` and can be updated in-place via
`record_score()` as the quality scorer produces judgements.

**Config toggle:**

| Field | Type | Default |
|-------|------|---------|
| `eviction_policy` | `str` | `"LRU"` (valid: `"LRU"`, `"LFU"`, `"FIFO"`, `"RR"`, `"COST_AWARE"`) |

**Prometheus metrics:**
- `byteai_eviction_cost_aware_evictions_total` — counter: total entries evicted under the cost-aware policy
- `byteai_eviction_cost_aware_savings_total` — counter: cumulative `∫ score × 1000` of evicted entries (proxy for preserved quality)

**Files:**
- `byte/manager/eviction/cost_aware.py` — `CostAwareCacheEviction`
- `byte/manager/eviction/memory_cache.py` — delegates policy `"COST_AWARE"` to the cost-aware class
- `byte/_core_lifecycle.py` — `_maybe_apply_cost_aware_eviction()` swaps the active policy post-init when the config requests it

---

## 16. arXiv 2503.05530 — Proximity: LSH Approximate Cache Prefilter

**Authors:** Shai Bergman, et al. (March 2025)
**Year:** 2025

**What it does:** A locality-sensitive hashing (MinHash-LSH) gate sits in front
of the expensive vector similarity search. Each cached query's text is shingled
and hashed into buckets; a new query's MinHash signature is probed against
those buckets in near-O(1) time. If a bucket hit arrives, the candidate IDs
are bubbled to the front of the vector search result list so the similarity
evaluator inspects them first. Reported 77.2% reduction in similarity-search
calls on the MedRAG workload with no recall loss.

ByteAI Cache's implementation uses the MIT-licensed `datasketch` library for
MinHash+LSH and hooks into the `SSDataManager` at two points:

1. **Write path** (`import_data`): every newly stored question is indexed in
   the LSH with its question_id as the key.
2. **Read path** (`search`): if the caller passed `question_text=...` as a
   kwarg, the LSH is probed first; matching IDs are prioritised in the
   returned result list.

**Config toggles:**

| Field | Type | Default |
|-------|------|---------|
| `lsh_prefilter_enabled` | `bool` | `False` |
| `lsh_num_perm` | `int` | `128` |
| `lsh_threshold` | `float` | `0.6` |
| `lsh_shingle_k` | `int` | `5` |

**Prometheus metrics:**
- `byteai_lsh_prefilter_lookups_total` — counter: total LSH probes
- `byteai_lsh_prefilter_tier0_hits_total` — counter: probes that found ≥1 near-duplicate
- `byteai_lsh_prefilter_skipped_searches_total` — counter: full vector searches avoided due to LSH short-circuit

**Files:**
- `byte/manager/lsh_prefilter.py` — `LSHPrefilter`, `get_lsh_prefilter()`, `reset_lsh_prefilter()`
- `byte/manager/data_manager.py` — LSH `index()` on insert, LSH `query()` + result reranking on search
