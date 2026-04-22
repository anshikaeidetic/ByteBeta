# ByteAI Benchmark Summary

- Generated at: `2026-04-19T15:44:14Z`
- Provider: `openai`
- Profile: `tier1`
- Live status: `skipped: missing OPENAI_API_KEY`

## Local Comparison

| system | phase | requests | hit_rate | p50_ms | p95_ms | cost_delta_pct | quality_pass_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| direct | cold | 6 | 0.00% | 0.06 | 0.11 | 0.00% | 100.00% |
| direct | warm_100 | 6 | 0.00% | 0.04 | 0.09 | 0.00% | 100.00% |
| native_cache | cold | 6 | 0.00% | 0.93 | 1.37 | 0.00% | 100.00% |
| native_cache | warm_100 | 6 | 0.00% | 0.66 | 0.85 | 0.00% | 100.00% |
| byte | cold | 6 | 0.00% | 2.04 | 10.89 | 85.45% | 100.00% |
| byte | warm_100 | 6 | 0.00% | 1.08 | 4.10 | 85.45% | 100.00% |

## Live Comparison

live skipped.
