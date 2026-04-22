from byte.benchmarking.corpus import FAMILY_SIZES, PROFILE_FAMILIES, load_profile, validate_profile
from byte.benchmarking.integrity import BENCHMARK_CONTRACT_VERSION, BENCHMARK_CORPUS_VERSION


def test_tier1_corpus_has_expected_family_counts_per_provider() -> None:
    profile = load_profile("tier1")
    summary = validate_profile(profile)

    for family in PROFILE_FAMILIES["tier1"]:
        expected = FAMILY_SIZES[family]
        counts = summary["family_counts"][family]
        assert counts["openai"] == expected
        assert counts["anthropic"] == expected
        assert counts["deepseek"] == expected
    assert summary["contract_versions"] == [BENCHMARK_CONTRACT_VERSION]


def test_tier1_corpus_can_be_filtered_by_provider_and_family_limit() -> None:
    profile = load_profile("tier1", providers=["deepseek"], max_items_per_family=7)
    summary = validate_profile(profile)

    assert profile["providers"] == ["deepseek"]
    for family, counts in summary["family_counts"].items():
        assert counts["deepseek"] == 7


def test_wrong_reuse_numeric_variants_track_prompt_price() -> None:
    profile = load_profile(
        "tier1",
        providers=["deepseek"],
        families=["wrong_reuse_detection"],
        max_items_per_family=20,
    )
    items = [item for item in profile["items"] if item.scenario == "near_miss_numeric"][:5]

    assert items
    expected_values = {str(item.expected_value) for item in items}
    assert len(expected_values) == len(items)
    for item in items:
        metadata = dict(item.metadata or {})
        prompt_text = str(item.input_payload["messages"][-1]["content"])
        assert str(metadata["price"]) in prompt_text


def test_prompt_distillation_profile_has_expected_family_counts_per_provider() -> None:
    profile = load_profile("prompt_distillation")
    summary = validate_profile(profile)

    for family in PROFILE_FAMILIES["prompt_distillation"]:
        expected = FAMILY_SIZES[family]
        counts = summary["family_counts"][family]
        assert counts["openai"] == expected
        assert counts["anthropic"] == expected
        assert counts["deepseek"] == expected


def test_provider_local_release_profile_uses_deepseek_only() -> None:
    profile = load_profile("tier1_v2_deepseek")

    assert profile["providers"] == ["deepseek"]
    assert profile["benchmark_track"] == "provider_local"
    assert profile["corpus_version"] == BENCHMARK_CORPUS_VERSION
