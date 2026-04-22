from byte.benchmarking.plans import openai_mixed_1000 as module


def test_workload_plan_hits_1000_requests_and_declares_video_gap() -> None:
    plan = module.build_workload_plan()

    assert plan["planned_request_count"] == 1000
    assert len(plan["text_items"]) == 800
    assert len(plan["moderation_inputs"]) == 80
    assert len(plan["image_prompts"]) == 40
    assert len(plan["speech_inputs"]) == 40
    assert plan["video_support"]["status"] == "unsupported"


def test_plain_unique_bucket_stays_unique_across_all_waves() -> None:
    plan = module.build_workload_plan()
    unique_prompts = [
        item["prompt"] for item in plan["text_items"] if item.get("bucket") == "plain_unique"
    ]

    assert len(unique_prompts) == 160
    assert len(set(unique_prompts)) == 160


def test_shared_context_sessions_are_unique_per_wave() -> None:
    plan = module.build_workload_plan()
    session_ids = [
        item["byte_context"]["byte_session_id"]
        for item in plan["text_items"]
        if item.get("bucket") == "shared_context_unique"
    ]

    assert len(session_ids) == 160
    assert len(set(session_ids)) == 40
    assert all(session_id.startswith("mega-session-") for session_id in session_ids)


def test_provider_coverage_matches_shared_capability_expectations() -> None:
    coverage = module.provider_coverage()

    assert coverage["openai"]["live_supported_request_count"] == 1000
    assert coverage["anthropic"]["live_supported_request_count"] == 800
    assert coverage["gemini"]["live_supported_request_count"] == 920
    assert coverage["groq"]["live_supported_request_count"] == 880
