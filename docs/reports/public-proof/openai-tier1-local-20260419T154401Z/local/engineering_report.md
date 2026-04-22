# Byte Benchmark Engineering Report

- Profile: tier1
- Benchmark track: platform_global
- Execution mode: partial
- Generated at: 2026-04-19T15:44:01
- Run id: tier1-20260419T154401Z-5665dac5
- Corpus version: byte-corpus-v2
- Benchmark contract version: byte-benchmark-v2
- Scoring version: byte-score-v3
- Trust policy version: byte-trust-v2
- Scorecard mode: dual
- Replicates: 1
- Confidence level: 0.95
- Judge mode: disabled
- Contamination check: False
- Live cutoff date: n/a
- Providers executed: openai
- Systems executed: byte, direct, native_cache

## Gate Summary

| Gate | Status | Value | Threshold |
| --- | --- | ---: | ---: |
| openai_false_reuse_rate | PASS | 0.0 | 0.01 |
| openai_fallback_trigger_rate | PASS | 1.0 | 0.95 |
| openai_confidence_score_accuracy | FAIL | 0.8333 | 0.9 |
| openai_confidence_ece | FAIL | 0.1715 | 0.08 |
| openai_deterministic_output_rate | PASS | 1.0 | 0.98 |
| openai_coverage_and_ci_present | PASS | 1 | 1 |
| openai_contamination_status_present | PASS | 1 | 1 |
| openai_perfect_accuracy_labeled | PASS | 1 | 1 |
| openai_scorecard_divergence_explained | PASS | 0.0 | 0.0 |
| openai_real_world_chaos_false_reuse_rate | PASS | 0.0 | 0.0025 |
| openai_wrong_reuse_detection_false_reuse_rate | PASS | 0.0 | 0.0025 |
| openai_degradation_unseen_false_reuse_rate | PASS | 0.0 | 0.01 |
| openai_real_world_chaos_accuracy_delta | PASS | 0.0 | -0.01 |
| openai_wrong_reuse_detection_accuracy_delta | PASS | 0.0 | 0.0 |
| openai_fuzzy_similarity_accuracy_delta | PASS | 0.0 | -0.01 |
| openai_generalization_accuracy_delta | PASS | 0.0 | -0.01 |
| openai_long_horizon_agents_accuracy_delta | PASS | 0.0 | -0.01 |
| openai_degradation_unseen_accuracy_delta | PASS | 0.0 | 0.0 |

## Provider Results

### openai

| System | Phase | Forced Accuracy | Forced CI 95 | Selective Accuracy | Coverage | Selective CI 95 | Avg Latency | Cost | False Reuse | Confidence ECE | Contamination |
| --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| direct | cold | 1.0000 | 0.6097-1.0000 | 1.0000 | 1.0000 | 0.6097-1.0000 | 0.06 | 0.000090 | 0.0000 | 0.6167 | mixed |
| direct | warm_100 | 1.0000 | 0.6097-1.0000 | 1.0000 | 1.0000 | 0.6097-1.0000 | 0.05 | 0.000090 | 0.0000 | 0.6167 | mixed |
| native_cache | cold | 1.0000 | 0.6097-1.0000 | 1.0000 | 1.0000 | 0.6097-1.0000 | 0.94 | 0.000079 | 0.0000 | 0.6167 | mixed |
| native_cache | warm_100 | 1.0000 | 0.6097-1.0000 | 1.0000 | 1.0000 | 0.6097-1.0000 | 0.65 | 0.000059 | 0.0000 | 0.6167 | mixed |
| byte | cold | 1.0000 | 0.6097-1.0000 | 1.0000 | 1.0000 | 0.6097-1.0000 | 4.09 | 0.000013 | 0.0000 | 0.1715 | mixed |
| byte | warm_100 | 1.0000 | 0.6097-1.0000 | 1.0000 | 1.0000 | 0.6097-1.0000 | 1.66 | 0.000013 | 0.0000 | 0.1715 | mixed |
