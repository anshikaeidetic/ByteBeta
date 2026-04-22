from byte import Config
from byte.processor.cache_latency import (
    clear_cache_stage_latency,
    record_cache_stage_outcome,
    should_bypass_cache_stage,
)


def setup_function() -> None:
    clear_cache_stage_latency()


def teardown_function() -> None:
    clear_cache_stage_latency()


def test_latency_guard_waits_for_probe_window_before_bypass() -> None:
    config = Config(
        enable_token_counter=False,
        cache_latency_min_samples=4,
        cache_latency_probe_samples=12,
        cache_latency_force_miss_samples=20,
        budget_latency_target_ms=100.0,
        cache_latency_p95_multiplier=1.0,
        cache_latency_min_hit_rate=0.2,
    )

    for _ in range(11):
        record_cache_stage_outcome("classification", "exact", latency_ms=180.0, hit=False)

    assert should_bypass_cache_stage("classification", "exact", config) is False


def test_latency_guard_waits_for_enough_hits_before_judging_hit_rate() -> None:
    config = Config(
        enable_token_counter=False,
        cache_latency_min_samples=4,
        cache_latency_probe_samples=12,
        cache_latency_min_hits=4,
        cache_latency_force_miss_samples=24,
        budget_latency_target_ms=100.0,
        cache_latency_p95_multiplier=1.0,
        cache_latency_min_hit_rate=0.4,
    )

    for index in range(12):
        record_cache_stage_outcome("classification", "exact", latency_ms=180.0, hit=index < 2)

    assert should_bypass_cache_stage("classification", "exact", config) is False


def test_latency_guard_can_bypass_after_sustained_zero_hit_tail_latency() -> None:
    config = Config(
        enable_token_counter=False,
        cache_latency_min_samples=4,
        cache_latency_probe_samples=12,
        cache_latency_force_miss_samples=16,
        budget_latency_target_ms=100.0,
        cache_latency_p95_multiplier=1.0,
        cache_latency_min_hit_rate=0.2,
    )

    for _ in range(16):
        record_cache_stage_outcome("question_answer", "semantic", latency_ms=185.0, hit=False)

    assert should_bypass_cache_stage("question_answer", "semantic", config) is True
