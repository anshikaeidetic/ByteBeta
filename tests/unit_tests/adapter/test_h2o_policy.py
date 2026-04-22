import pytest

from byte.h2o.policy import H2OSequenceCache, resolve_h2o_settings

torch = pytest.importorskip("torch")


def test_resolve_h2o_settings_rejects_unsupported_family() -> None:
    settings = resolve_h2o_settings(
        enabled=True,
        prompt_tokens=32,
        model_family="unknown",
        heavy_ratio=0.2,
        recent_ratio=0.2,
    )

    assert settings.requested is True
    assert settings.applied is False
    assert settings.fallback_reason == "unsupported_model_family"


def test_h2o_policy_keeps_heavy_hitters_and_recent_tokens() -> None:
    settings = resolve_h2o_settings(
        enabled=True,
        prompt_tokens=10,
        model_family="llama",
        heavy_ratio=0.2,
        recent_ratio=0.2,
    )
    policy = H2OSequenceCache(settings)
    attention = torch.zeros((1, 4, 10, 10), dtype=torch.float32)
    attention[:, :, :, 1] = 100.0
    attention[:, :, :, 3] = 90.0
    key_states = torch.zeros((1, 4, 10, 1), dtype=torch.float32)
    value_states = torch.zeros((1, 4, 10, 1), dtype=torch.float32)
    for index in range(10):
        key_states[:, :, index, 0] = float(index)
        value_states[:, :, index, 0] = float(index)

    past_key_values = ((key_states, value_states),)
    cropped, stats = policy.apply(past_key_values, (attention,))

    assert stats["retained_tokens"] == 4
    assert stats["evicted_tokens"] == 6
    assert cropped[0][0].shape[2] == 4
    kept_positions = cropped[0][0][0, 0, :, 0].tolist()
    assert kept_positions == [1.0, 3.0, 8.0, 9.0]


def test_h2o_policy_aggregates_attention_heads_to_kv_heads() -> None:
    settings = resolve_h2o_settings(
        enabled=True,
        prompt_tokens=8,
        model_family="mistral",
        heavy_ratio=0.25,
        recent_ratio=0.25,
    )
    policy = H2OSequenceCache(settings)
    attention = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    attention[:, 0:2, :, 2] = 50.0
    attention[:, 2:4, :, 4] = 40.0
    key_states = torch.zeros((1, 2, 8, 1), dtype=torch.float32)
    value_states = torch.zeros((1, 2, 8, 1), dtype=torch.float32)
    for index in range(8):
        key_states[:, :, index, 0] = float(index)
        value_states[:, :, index, 0] = float(index)

    cropped, stats = policy.apply(((key_states, value_states),), (attention,))

    assert stats["retained_tokens"] == 4
    assert cropped[0][0].shape[1] == 2
    first_head_positions = cropped[0][0][0, 0, :, 0].tolist()
    second_head_positions = cropped[0][0][0, 1, :, 0].tolist()
    assert first_head_positions == [2.0, 4.0, 6.0, 7.0]
    assert second_head_positions == [2.0, 4.0, 6.0, 7.0]


def test_h2o_policy_skips_evicting_short_sequences() -> None:
    settings = resolve_h2o_settings(
        enabled=True,
        prompt_tokens=4,
        model_family="opt",
        heavy_ratio=0.5,
        recent_ratio=0.5,
    )
    policy = H2OSequenceCache(settings)
    attention = torch.ones((1, 4, 4, 4), dtype=torch.float32)
    key_states = torch.randn((1, 4, 4, 2), dtype=torch.float32)
    value_states = torch.randn((1, 4, 4, 2), dtype=torch.float32)

    cropped, stats = policy.apply(((key_states, value_states),), (attention,))

    assert stats["evicted_tokens"] == 0
    assert torch.equal(cropped[0][0], key_states)
    assert torch.equal(cropped[0][1], value_states)


def test_h2o_policy_reset_clears_layer_state() -> None:
    settings = resolve_h2o_settings(
        enabled=True,
        prompt_tokens=8,
        model_family="gpt_neox",
        heavy_ratio=0.25,
        recent_ratio=0.25,
    )
    policy = H2OSequenceCache(settings)
    attention = torch.ones((1, 4, 8, 8), dtype=torch.float32)
    key_states = torch.randn((1, 4, 8, 2), dtype=torch.float32)
    value_states = torch.randn((1, 4, 8, 2), dtype=torch.float32)

    policy.apply(((key_states, value_states),), (attention,))
    policy.reset()
    cropped, stats = policy.apply(((key_states, value_states),), (attention,))

    assert stats["retained_tokens"] == 4
    assert len(policy._layers) == 1
