from byte.benchmarking.plans import deepseek_reasoning_reuse as module


def test_reasoning_reuse_plan_hits_600_requests_with_6_waves() -> None:
    plan = module.build_workload_plan()

    assert plan["planned_request_count"] == 600
    assert len(plan["waves"]) == 6
    assert all(len(wave["items"]) == 100 for wave in plan["waves"])
    assert len(plan["items"]) == 600
    assert len(plan["warmup_items"]) == 30


def test_reasoning_reuse_distribution_matches_expected_totals() -> None:
    plan = module.build_workload_plan()

    assert plan["distribution"] == {
        "complex_reasoning_chain": 200,
        "multi_step_workflows": 200,
        "repeated_knowledge_queries": 200,
    }


def test_reasoning_reuse_plan_uses_single_deepseek_model_and_expected_workload_kinds() -> None:
    plan = module.build_workload_plan()

    assert {item["model"] for item in plan["items"]} == {"deepseek-chat"}
    assert {item["kind"] for item in plan["items"]} == {
        "profit_margin",
        "refund_policy",
        "capital_city",
    }


def test_reasoning_reuse_plan_seeds_all_knowledge_groups_before_reuse_variants() -> None:
    plan = module.build_workload_plan()

    wave_one_knowledge = [
        item
        for item in plan["waves"][0]["items"]
        if item["scenario"] == "repeated_knowledge_queries"
    ]
    assert len(wave_one_knowledge) == 20
    assert {item["variant"] for item in wave_one_knowledge} == {"v01"}
