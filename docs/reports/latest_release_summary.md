# Latest Release Summary

## Run Metadata

- Run ID: `tier1_v2_deepseek-20260401T143351Z-3e1bcf1b`
- Generated at: `2026-04-01T14:33:51`
- Profile: `tier1_v2_deepseek`
- Benchmark track: `provider_local`
- Execution mode: `partial`
- Providers executed: `deepseek`
- Systems executed: `byte`

## Gate Results

The latest retained benchmark artifact is a targeted growth hardening run. It is useful as a regression check, but it is not a full release sign-off.

## Highlights

- No false reuse was observed across the retained DeepSeek growth hardening lane.
- Calibration and deterministic-output gates cleared their thresholds comfortably.
- The report-level release gate remained `false` because this run executed in `partial` mode.
- The metrics below come from the retained internal checkpoint artifact for this lane and are not offered as a third-party reproducibility bundle.

- False reuse rate: `0.0000` against a `0.0100` threshold
- Fallback trigger rate: `1.0000` against a `0.9500` threshold
- Confidence score accuracy: `0.9722` against a `0.9000` threshold
- Confidence ECE: `0.0572` against a `0.0800` threshold
- Deterministic output rate: `1.0000` against a `0.9800` threshold
- Coverage, contamination labeling, perfect-accuracy labeling, and divergence explanation checks all passed
- Real-world chaos, wrong reuse detection, and degradation-unseen false reuse checks all passed at `0.0000`

## Dollar Impact

The latest partial hardening run did not emit a dollar-impact payload. Cost modeling remains available in the broader benchmark system, but it is not represented in this retained summary artifact.

## Readiness Interpretation

This run shows the current hardening path is behaving as intended on the targeted DeepSeek growth benchmark lane. It should be treated as a focused engineering checkpoint, not as a complete release gate, because the retained artifact was produced in partial execution mode and the report-level release gate remained `false`.
