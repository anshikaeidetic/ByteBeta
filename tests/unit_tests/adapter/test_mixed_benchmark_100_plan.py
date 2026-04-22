from byte.benchmarking.plans import openai_mixed_100 as module


def test_workload_plan_hits_100_requests_and_declares_video_gap() -> None:
    plan = module.build_workload_plan()

    assert plan["planned_request_count"] == 100
    assert len(plan["text_items"]) == 60
    assert len(plan["moderation_inputs"]) == 10
    assert len(plan["image_prompts"]) == 10
    assert len(plan["speech_inputs"]) == 10
    assert plan["video_support"]["status"] == "unsupported"


def test_workload_plan_keeps_unique_prompt_buckets() -> None:
    plan = module.build_workload_plan()

    plain_unique = [item for item in plan["text_items"] if item.get("bucket") == "plain_unique"]
    shared_context = [
        item for item in plan["text_items"] if item.get("bucket") == "shared_context_unique"
    ]

    assert len(plain_unique) == 12
    assert len(shared_context) == 12
    assert len({item["prompt"] for item in plain_unique}) == 12


def test_provider_coverage_matches_100_request_capability_expectations() -> None:
    coverage = module.provider_coverage()

    assert coverage["openai"]["live_supported_request_count"] == 100
    assert coverage["anthropic"]["live_supported_request_count"] == 60
    assert coverage["gemini"]["live_supported_request_count"] == 90
    assert coverage["groq"]["live_supported_request_count"] == 80
    assert coverage["openrouter"]["live_supported_request_count"] == 70
    assert coverage["ollama"]["live_supported_request_count"] == 60
    assert coverage["mistral"]["live_supported_request_count"] == 60
    assert coverage["cohere"]["live_supported_request_count"] == 60
    assert coverage["bedrock"]["live_supported_request_count"] == 60
