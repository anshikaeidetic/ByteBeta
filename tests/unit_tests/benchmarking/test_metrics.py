from byte.benchmarking.metrics import (
    compare_summaries,
    expected_calibration_error,
    mean_confidence_interval,
    proportion_confidence_interval,
    summarize_records,
)


def test_summarize_records_computes_new_trust_metrics() -> None:
    records = [
        {
            "status_code": 200,
            "latency_ms": 120.0,
            "output_correct": True,
            "reuse_safe": True,
            "must_fallback": False,
            "actual_reuse": True,
            "fallback_taken": False,
            "reuse_confidence": 0.95,
            "policy_adherent": True,
            "deterministic_expected": True,
            "deterministic_output": True,
            "workflow_steps_skipped": 6,
            "workflow_total_steps": 6,
            "upstream_calls": 0,
            "served_via": "reuse",
            "tokens": {"prompt_tokens": 0, "cached_prompt_tokens": 0, "completion_tokens": 0},
            "original_prompt_tokens": 200,
            "distilled_prompt_tokens": 80,
            "original_prompt_chars": 800,
            "distilled_prompt_chars": 320,
            "faithfulness_pass": True,
            "faithfulness_score": 1.0,
            "entity_preservation_rate": 1.0,
            "schema_preservation_rate": 1.0,
            "module_hits": 2,
            "distillation_fallback": False,
            "cost_usd": 0.0,
        },
        {
            "status_code": 200,
            "latency_ms": 400.0,
            "output_correct": True,
            "reuse_safe": False,
            "must_fallback": True,
            "actual_reuse": False,
            "fallback_taken": True,
            "reuse_confidence": 0.05,
            "policy_adherent": True,
            "deterministic_expected": True,
            "deterministic_output": True,
            "workflow_steps_skipped": 0,
            "workflow_total_steps": 6,
            "upstream_calls": 1,
            "served_via": "upstream",
            "tokens": {"prompt_tokens": 120, "cached_prompt_tokens": 0, "completion_tokens": 8},
            "original_prompt_tokens": 120,
            "distilled_prompt_tokens": 120,
            "original_prompt_chars": 480,
            "distilled_prompt_chars": 480,
            "faithfulness_pass": True,
            "faithfulness_score": 1.0,
            "entity_preservation_rate": 1.0,
            "schema_preservation_rate": 1.0,
            "module_hits": 0,
            "distillation_fallback": False,
            "cost_usd": 0.001,
        },
    ]
    summary = summarize_records(records)

    assert summary["accuracy_ratio"] == 1.0
    assert summary["forced_answer_accuracy"] == 1.0
    assert summary["selective_accuracy"] == 1.0
    assert summary["coverage"] == 1.0
    assert summary["false_reuse_rate"] == 0.0
    assert summary["fallback_trigger_rate"] == 1.0
    assert summary["safe_reuse_recall"] == 1.0
    assert summary["workflow_step_reduction"] == 0.5
    assert summary["confidence_score_accuracy"] == 1.0
    assert summary["prompt_token_reduction_ratio"] == 0.375
    assert summary["faithfulness_pass_rate"] == 1.0
    assert summary["module_reuse_rate"] == 0.5
    assert "ci_95" in summary


def test_expected_calibration_error_and_summary_comparison() -> None:
    records = [
        {"reuse_confidence": 0.9, "reuse_safe": True},
        {"reuse_confidence": 0.8, "reuse_safe": True},
        {"reuse_confidence": 0.2, "reuse_safe": False},
        {"reuse_confidence": 0.1, "reuse_safe": False},
    ]
    assert expected_calibration_error(records) < 0.2

    baseline = {"cost_usd": 10.0, "avg_latency_ms": 1000.0, "accuracy_ratio": 0.8, "tokens": {"prompt_tokens": 1000}}
    summary = {"cost_usd": 4.0, "avg_latency_ms": 400.0, "accuracy_ratio": 0.9, "tokens": {"prompt_tokens": 200}}
    comparison = compare_summaries(summary, baseline)

    assert comparison["cost_reduction_ratio"] == 0.6
    assert comparison["latency_improvement_ratio"] == 0.6
    assert comparison["accuracy_delta"] == 0.1
    assert comparison["token_reduction_ratio"] == 0.8
    assert comparison["coverage_delta"] == 0.0


def test_dual_scorecard_tracks_coverage_and_intervals() -> None:
    summary = summarize_records(
        [
            {
                "status_code": 200,
                "response_text": "ALLOW",
                "output_correct": True,
                "policy_adherent": True,
                "reuse_safe": True,
                "reuse_confidence": 0.9,
                "tokens": {"prompt_tokens": 0, "cached_prompt_tokens": 0, "completion_tokens": 0},
            },
            {
                "status_code": 200,
                "response_text": "",
                "answered": False,
                "abstained": True,
                "output_correct": False,
                "policy_adherent": True,
                "reuse_safe": False,
                "must_fallback": True,
                "reuse_confidence": 0.1,
                "tokens": {"prompt_tokens": 0, "cached_prompt_tokens": 0, "completion_tokens": 0},
            },
        ]
    )

    assert summary["forced_answer_accuracy"] == 0.5
    assert summary["selective_accuracy"] == 1.0
    assert summary["coverage"] == 0.5
    assert summary["scorecard_explanation"]
    assert summary["ci_95"]["coverage"]["high"] >= summary["ci_95"]["coverage"]["low"]


def test_confidence_interval_helpers_are_deterministic() -> None:
    proportion_ci = proportion_confidence_interval(9, 10)
    mean_ci = mean_confidence_interval([0.8, 0.9, 1.0])

    assert proportion_ci["high"] >= proportion_ci["low"]
    assert mean_ci["count"] == 3
    assert mean_ci["high"] >= mean_ci["low"]
