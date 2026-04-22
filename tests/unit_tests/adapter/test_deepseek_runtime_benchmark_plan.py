from byte.benchmarking.plans import deepseek_runtime_optimization as module


def test_workload_plan_hits_1200_requests_with_12_waves() -> None:
    plan = module.build_workload_plan()

    assert plan["planned_request_count"] == 1200
    assert len(plan["waves"]) == 12
    assert all(len(wave["items"]) == 100 for wave in plan["waves"])
    assert len(plan["items"]) == 1200
    assert len(plan["warmup_items"]) == 50


def test_workload_distribution_matches_corrected_total() -> None:
    plan = module.build_workload_plan()

    assert plan["distribution"] == {
        "exact_repeat": 150,
        "normalized_variant": 150,
        "plain_unique": 150,
        "shared_context_unique": 250,
        "rag_queries": 200,
        "coding_tasks": 150,
        "agent_workflows": 100,
        "long_context": 50,
    }


def test_coding_lane_uses_deepseek_coder_and_contextual_workloads_have_sessions() -> None:
    plan = module.build_workload_plan()

    coding_items = [item for item in plan["items"] if item["scenario"] == "coding_tasks"]
    assert coding_items
    assert all(item["model"] == "deepseek-coder" for item in coding_items)

    contextual_items = [item for item in plan["items"] if item.get("request_style") == "contextual"]
    assert contextual_items
    assert all("byte_session_id" in item["byte_context"] for item in contextual_items)
