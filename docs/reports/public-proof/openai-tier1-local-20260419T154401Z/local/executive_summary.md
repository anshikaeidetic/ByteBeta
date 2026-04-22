# Byte Benchmark Executive Summary

Profile: tier1
Benchmark track: platform_global
Execution mode: partial
Corpus version: byte-corpus-v2
Benchmark contract version: byte-benchmark-v2
Scoring version: byte-score-v3
Trust policy version: byte-trust-v2
Scorecard mode: dual

## Headline Findings

- openai: Byte forced accuracy 1.0000 (95% CI 0.6097-1.0000), selective accuracy 1.0000 at coverage 1.0000.
- openai: Byte cost reduction 0.8545, latency improvement -32.2000, forced accuracy delta 0.0000.
- openai: contamination mixed, false reuse rate 0.0000, fallback trigger rate 1.0000, confidence accuracy 0.8333.
- openai: prompt reduction 0.0000, faithfulness 0.0000, module reuse 0.0000.

## DeepSeek Scorecards

| Provider | Phase | Scorecard | Accuracy | Coverage | Sample Size | CI 95 | Contamination |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| openai | warm_100 | Forced | 1.0000 | 1.0000 | 6 | 0.6097-1.0000 | mixed |
| openai | warm_100 | Selective | 1.0000 | 1.0000 | 6 | 0.6097-1.0000 | mixed |

## Prompt Distillation

| Provider | Prompt Reduction | Faithfulness | Module Reuse |
| --- | ---: | ---: | ---: |
| openai | 0.0000 | 0.0000 | 0.0000 |

## Dollar Impact

| Volume | Direct | Byte | Savings |
| --- | ---: | ---: | ---: |
| 100k | 1.50 | 0.22 | 1.28 |
| 1m | 15.02 | 2.19 | 12.84 |
| 10m | 150.25 | 21.87 | 128.38 |
