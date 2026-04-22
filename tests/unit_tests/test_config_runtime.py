from byte import Config
from byte.config import CacheConfig


def test_explicit_flat_override_beats_env_and_section_default(monkeypatch) -> None:
    monkeypatch.setenv("BYTE_SIMILARITY_THRESHOLD", "0.73")

    cfg = Config(
        cache=CacheConfig(similarity_threshold=0.61),
        similarity_threshold=0.91,
    )

    assert cfg.similarity_threshold == 0.91
    assert cfg.cache.similarity_threshold == 0.91


def test_env_override_beats_section_default_when_no_flat_override(monkeypatch) -> None:
    monkeypatch.setenv("BYTE_SIMILARITY_THRESHOLD", "0.84")

    cfg = Config(cache=CacheConfig(similarity_threshold=0.61))

    assert cfg.similarity_threshold == 0.84
    assert cfg.cache.similarity_threshold == 0.84
