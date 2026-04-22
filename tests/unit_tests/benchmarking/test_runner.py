import json
from pathlib import Path

from byte.benchmarking.integrity import (
    BENCHMARK_CONTRACT_VERSION,
    BENCHMARK_SCORING_VERSION,
    BENCHMARK_TRUST_POLICY_VERSION,
)
from byte.benchmarking.reporting import build_engineering_report
from byte.benchmarking.runner import run_suite


class _FakeSystem:
    def __init__(self, provider: str, system: str) -> None:
        self.spec = type(
            "Spec",
            (),
            {
                "provider": provider,
                "system": system,
                "baseline_type": "byte" if system == "byte" else "direct",
            },
        )()
        self.warmup_item_ids = []

    def is_available(self) -> object:
        return True, ""

    def begin_phase(self, warmup_items) -> None:
        self._warmed = len(warmup_items)
        self.warmup_item_ids = [item.item_id for item in warmup_items]

    def end_phase(self) -> None:
        return None

    def run_item(self, item, *, phase) -> object:
        is_byte = self.spec.system == "byte"
        served_via = "reuse" if is_byte and item.reuse_safe else "upstream"
        fallback_taken = bool(item.must_fallback) if is_byte else False
        actual_reuse = served_via == "reuse"
        return {
            "system": self.spec.system,
            "provider": self.spec.provider,
            "baseline_type": self.spec.baseline_type,
            "phase": phase,
            "family": item.family,
            "scenario": item.scenario,
            "seed_id": item.seed_id,
            "variant_id": item.variant_id,
            "reuse_safe": item.reuse_safe,
            "must_fallback": item.must_fallback,
            "actual_reuse": actual_reuse,
            "fallback_taken": fallback_taken,
            "reuse_confidence": 0.95 if actual_reuse else 0.05,
            "reuse_decision_correct": True,
            "output_correct": True,
            "policy_adherent": True,
            "deterministic_output": True,
            "deterministic_expected": item.deterministic_expected,
            "workflow_steps_skipped": item.workflow_total_steps if is_byte else 0,
            "workflow_total_steps": item.workflow_total_steps,
            "upstream_calls": 0 if is_byte else 1,
            "tokens": {"prompt_tokens": 0 if is_byte else 10, "cached_prompt_tokens": 0, "completion_tokens": 0},
            "original_prompt_tokens": 100,
            "distilled_prompt_tokens": 40 if is_byte else 100,
            "original_prompt_chars": 400,
            "distilled_prompt_chars": 160 if is_byte else 400,
            "prompt_token_reduction_ratio": 0.6 if is_byte else 0.0,
            "compression_ratio": 0.6 if is_byte else 0.0,
            "faithfulness_pass": True,
            "faithfulness_score": 1.0,
            "entity_preservation_rate": 1.0,
            "schema_preservation_rate": 1.0,
            "module_hits": 2 if is_byte else 0,
            "distillation_fallback": False,
            "prompt_distillation_applied": is_byte,
            "cost_usd": 0.0 if is_byte else 0.001,
            "status_code": 200,
            "latency_ms": 25.0 if is_byte else 125.0,
            "trace_ref": f"{self.spec.provider}/{self.spec.system}/{phase}/{item.item_id}",
            "served_via": served_via,
            "byte_reason": "",
            "response_text": str(item.expected_value),
            "canonical_output": str(item.expected_value),
            "tags": list(item.tags),
            "item_id": item.item_id,
            "provider_model": "test-model",
            "configured_model": "test-model",
            "error": "",
        }


def test_run_suite_writes_report_bundle(monkeypatch, tmp_path) -> object:
    def _build_systems(providers, systems) -> object:
        return [_FakeSystem("deepseek", "direct"), _FakeSystem("deepseek", "byte")]

    monkeypatch.setattr("byte.benchmarking.runner.build_systems", _build_systems)
    results = run_suite(
        profile="tier1_v2_deepseek",
        providers=["deepseek"],
        systems=["direct", "byte"],
        phases=["cold", "warm_100"],
        out_dir=str(tmp_path),
        max_items_per_family=2,
        concurrency=1,
        fail_on_thresholds=False,
        scorecard_mode="dual",
        replicates=2,
    )

    assert results["providers_executed"] == ["deepseek"]
    assert sorted(results["systems_executed"]) == ["byte", "direct"]
    assert results["benchmark_track"] == "provider_local"
    engineering_json = Path(results["artifacts"]["engineering_json"])
    engineering_details_json = Path(results["artifacts"]["engineering_details_json"])
    assert engineering_json.exists()
    assert engineering_details_json.exists()
    assert Path(results["artifacts"]["engineering_markdown"]).exists()
    assert Path(results["artifacts"]["executive_markdown"]).exists()
    assert "deepseek" in results["providers"]
    assert results["benchmark_contract_version"] == BENCHMARK_CONTRACT_VERSION
    assert results["scoring_version"] == BENCHMARK_SCORING_VERSION
    assert results["trust_policy_version"] == BENCHMARK_TRUST_POLICY_VERSION
    summary = results["providers"]["deepseek"]["systems"]["byte"]["phases"]["warm_100"]["summary"]
    assert "forced_answer_accuracy" in summary
    assert "selective_accuracy" in summary
    assert "ci_95" in summary
    assert len(results["providers"]["deepseek"]["systems"]["byte"]["phases"]["warm_100"]["replicates"]) == 2
    engineering_report = json.loads(engineering_json.read_text(encoding="utf-8"))
    assert sorted(engineering_report) == [
        "benchmark_contract_version",
        "benchmark_track",
        "confidence_level",
        "contamination_check",
        "corpus_version",
        "execution_mode",
        "gate_results",
        "generated_at",
        "judge_mode",
        "live_cutoff_date",
        "manifest_versions",
        "phase_summaries",
        "profile",
        "profile_base",
        "providers_executed",
        "providers_requested",
        "records",
        "release_gate",
        "replicates",
        "run_id",
        "schema_version",
        "scorecard_mode",
        "scoring_version",
        "systems_executed",
        "systems_requested",
        "trust_policy_version",
    ]
    expected_records = 0
    expected_phase_summaries = 0
    for provider_payload in results["providers"].values():
        for system_payload in provider_payload["systems"].values():
            for phase_payload in system_payload["phases"].values():
                expected_phase_summaries += 1
                expected_records += len(phase_payload["records"])
    assert len(engineering_report["records"]) == expected_records
    assert len(engineering_report["phase_summaries"]) == expected_phase_summaries
    assert build_engineering_report(engineering_report) == engineering_report


def test_run_suite_warms_with_disjoint_pool(monkeypatch, tmp_path) -> object:
    byte_system = _FakeSystem("deepseek", "byte")

    def _build_systems(providers, systems) -> object:
        return [byte_system]

    monkeypatch.setattr("byte.benchmarking.runner.build_systems", _build_systems)
    results = run_suite(
        profile="tier1_v2_deepseek",
        providers=["deepseek"],
        systems=["byte"],
        phases=["warm_100"],
        out_dir=str(tmp_path),
        max_items_per_family=2,
        concurrency=1,
        fail_on_thresholds=False,
    )

    scored_ids = {
        record["item_id"]
        for record in results["providers"]["deepseek"]["systems"]["byte"]["phases"]["warm_100"]["records"]
    }
    assert byte_system.warmup_item_ids
    assert not scored_ids.intersection(byte_system.warmup_item_ids)


def test_prompt_distillation_profile_emits_prompt_metrics(monkeypatch, tmp_path) -> object:
    def _build_systems(providers, systems) -> object:
        return [_FakeSystem("deepseek", "direct"), _FakeSystem("deepseek", "byte")]

    monkeypatch.setattr("byte.benchmarking.runner.build_systems", _build_systems)
    results = run_suite(
        profile="prompt_distillation_v2_deepseek",
        providers=["deepseek"],
        systems=["direct", "byte"],
        phases=["warm_100"],
        out_dir=str(tmp_path),
        max_items_per_family=2,
        concurrency=1,
        fail_on_thresholds=False,
    )

    summary = results["providers"]["deepseek"]["systems"]["byte"]["phases"]["warm_100"]["summary"]
    assert summary["prompt_token_reduction_ratio"] > 0.0
    assert summary["faithfulness_pass_rate"] == 1.0


def test_release_reports_use_labeled_scorecards(monkeypatch, tmp_path) -> object:
    def _build_systems(providers, systems) -> object:
        return [_FakeSystem("deepseek", "direct"), _FakeSystem("deepseek", "byte")]

    monkeypatch.setattr("byte.benchmarking.runner.build_systems", _build_systems)
    results = run_suite(
        profile="tier1_v2_deepseek",
        providers=["deepseek"],
        systems=["direct", "byte"],
        phases=["warm_100"],
        out_dir=str(tmp_path),
        max_items_per_family=2,
        concurrency=1,
        fail_on_thresholds=False,
    )

    executive = Path(results["artifacts"]["executive_markdown"]).read_text(encoding="utf-8")
    assert "Forced" in executive
    assert "Selective" in executive
    assert "| Provider | Phase | Scorecard | Accuracy | Coverage | Sample Size | CI 95 | Contamination |" in executive
