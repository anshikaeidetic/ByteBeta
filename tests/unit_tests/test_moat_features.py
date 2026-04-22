"""Unit tests for the 4 moat features:
1. Conversation Fingerprinting
2. Cache Warming
3. Multi-Tier Cascading Cache
4. Cache Quality Scoring
"""

import pytest

# ──────────────────────────────────────────────────────────────────
# 1. Conversation Fingerprinting
# ──────────────────────────────────────────────────────────────────
from byte.processor.fingerprint import ConversationFingerprinter


class TestConversationFingerprinter:
    def setup_method(self) -> None:
        self.fp = ConversationFingerprinter(window_size=3, decay_factor=0.5)

    def test_empty_messages(self) -> None:
        assert self.fp.fingerprint([]) == ""

    def test_single_message_fingerprint(self) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        result = self.fp.fingerprint(msgs)
        assert isinstance(result, str)
        assert len(result) == 64  # sha256 hex digest

    def test_different_contexts_different_fingerprints(self) -> None:
        """Two 'Tell me more' with different history must produce different FPs."""
        msgs_a = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a language."},
            {"role": "user", "content": "Tell me more"},
        ]
        msgs_b = [
            {"role": "user", "content": "What is Rust?"},
            {"role": "assistant", "content": "Rust is a language."},
            {"role": "user", "content": "Tell me more"},
        ]
        assert self.fp.fingerprint(msgs_a) != self.fp.fingerprint(msgs_b)

    def test_same_context_same_fingerprint(self) -> None:
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "how are you?"},
        ]
        assert self.fp.fingerprint(msgs) == self.fp.fingerprint(msgs)

    def test_context_key_empty_for_single_turn(self) -> None:
        assert self.fp.context_key([{"role": "user", "content": "hi"}]) == ""

    def test_context_key_returns_16_chars(self) -> None:
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        key = self.fp.context_key(msgs)
        assert len(key) == 16

    def test_enrich_pre_embedding_single_turn(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        result = self.fp.enrich_pre_embedding("hi", msgs)
        assert result == "hi"  # no context for single turn

    def test_enrich_pre_embedding_multi_turn(self) -> None:
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        result = self.fp.enrich_pre_embedding("c", msgs)
        assert result.startswith("c||ctx:")

    def test_role_matters(self) -> None:
        """Same text from different roles should produce different fingerprints."""
        msgs_user = [{"role": "user", "content": "hello"}]
        msgs_asst = [{"role": "assistant", "content": "hello"}]
        assert self.fp.fingerprint(msgs_user) != self.fp.fingerprint(msgs_asst)

    def test_window_respects_size(self) -> None:
        fp_small = ConversationFingerprinter(window_size=1)
        fp_large = ConversationFingerprinter(window_size=5)
        msgs = [{"role": "user", "content": f"msg{i}"} for i in range(5)]
        # Small window only sees last message, so adding prefix msgs doesn't change FP
        result_small = fp_small.fingerprint(msgs)
        result_large = fp_large.fingerprint(msgs)
        # They see different amounts of context, so FPs differ
        assert result_small != result_large


# ──────────────────────────────────────────────────────────────────
# 2. Multi-Tier Cascading Cache
# ──────────────────────────────────────────────────────────────────
from byte.manager.tiered_cache import TieredCacheManager


class MockBackend:
    """Minimal backend that stores data in a dict."""

    def __init__(self) -> None:
        self.store = {}

    def save(self, question, answer, embedding_data, **kwargs) -> None:
        key = embedding_data.tobytes() if hasattr(embedding_data, "tobytes") else embedding_data
        self.store[key] = (question, answer)

    def search(self, embedding_data, **kwargs) -> object:
        key = embedding_data.tobytes() if hasattr(embedding_data, "tobytes") else embedding_data
        if key in self.store:
            return [self.store[key]]
        return [None]

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class TestTieredCacheManager:
    def setup_method(self) -> None:
        self.backend = MockBackend()
        self.tcm = TieredCacheManager(
            self.backend,
            tier1_max_size=3,
            promotion_threshold=2,
            promotion_window_s=60.0,
        )

    def test_save_delegates_to_backend(self) -> None:
        self.tcm.save("q", "a", b"emb1")
        assert b"emb1" in self.backend.store

    def test_search_tier2_first_access(self) -> None:
        self.backend.store[b"emb1"] = ("q", "a")
        result = self.tcm.search(b"emb1")
        assert result == [("q", "a")]
        stats = self.tcm.stats()
        assert stats["tier2_hits"] == 1
        assert stats["tier1_hits"] == 0

    def test_promotion_after_threshold_accesses(self) -> None:
        self.backend.store[b"emb1"] = ("q", "a")
        # First access: tier2 hit, 1 access recorded (below threshold of 2)
        self.tcm.search(b"emb1")
        assert self.tcm.stats()["promotions"] == 0
        # Second access: tier2 hit, 2 accesses -> promoted
        self.tcm.search(b"emb1")
        assert self.tcm.stats()["promotions"] == 1
        # Third access: tier1 hit
        self.tcm.search(b"emb1")
        assert self.tcm.stats()["tier1_hits"] == 1

    def test_tier1_eviction_when_full(self) -> None:
        for i in range(4):
            key = f"emb{i}".encode()
            self.backend.store[key] = (f"q{i}", f"a{i}")
            # Access twice to promote
            self.tcm.search(key)
            self.tcm.search(key)
        stats = self.tcm.stats()
        assert stats["tier1_size"] <= 3
        assert stats["demotions"] >= 1

    def test_miss_tracked(self) -> None:
        self.tcm.search(b"nonexistent")
        assert self.tcm.stats()["misses"] == 1

    def test_stats_returns_all_fields(self) -> None:
        stats = self.tcm.stats()
        required_fields = [
            "tier1_size",
            "tier1_max_size",
            "tier1_hits",
            "tier2_hits",
            "misses",
            "total_requests",
            "tier1_hit_ratio",
            "promotions",
            "demotions",
        ]
        for field in required_fields:
            assert field in stats


# ──────────────────────────────────────────────────────────────────
# 3. Cache Quality Scoring
# ──────────────────────────────────────────────────────────────────
from byte.processor.quality import QualityScorer


class TestQualityScorer:
    def setup_method(self) -> None:
        self.scorer = QualityScorer(
            auto_evict_threshold=0.2,
            feedback_weight=0.1,
        )

    def test_high_similarity_high_score(self) -> None:
        score = self.scorer.score("What is Python?", "Python is a language.", 0.95)
        assert score > 0.5

    def test_low_similarity_low_score(self) -> None:
        score = self.scorer.score("What is Python?", "Hello.", 0.1)
        assert score < 0.5

    def test_feedback_positive_increases_score(self) -> None:
        self.scorer.score("test query", "test answer", 0.5)
        result = self.scorer.record_feedback("test query", thumbs_up=True)
        assert result["new_score"] > 0.0

    def test_feedback_negative_decreases_score(self) -> None:
        self.scorer.score("test query", "test answer", 0.5)
        initial = self.scorer._entries[self.scorer._hash("test query")]["score"]
        self.scorer.record_feedback("test query", thumbs_up=False)
        after = self.scorer._entries[self.scorer._hash("test query")]["score"]
        assert after < initial

    def test_auto_evict_detection(self) -> None:
        # Start with a very low score
        self.scorer.score("bad query", "x", 0.05)
        # Pile on negative feedback
        for _ in range(5):
            self.scorer.record_feedback("bad query", thumbs_up=False)
        assert self.scorer.should_evict("bad query")

    def test_good_query_not_evicted(self) -> None:
        self.scorer.score("good query", "Great detailed answer here.", 0.95)
        self.scorer.record_feedback("good query", thumbs_up=True)
        assert not self.scorer.should_evict("good query")

    def test_get_low_quality_entries(self) -> None:
        self.scorer.score("good", "answer", 0.9)
        self.scorer.score("bad", "x", 0.05)
        for _ in range(5):
            self.scorer.record_feedback("bad", thumbs_up=False)
        low = self.scorer.get_low_quality_entries()
        assert any(e["query_preview"].startswith("bad") for e in low)

    def test_stats_returns_all_fields(self) -> None:
        self.scorer.score("q", "a", 0.5)
        stats = self.scorer.stats()
        required = [
            "total_entries_tracked",
            "total_scored",
            "total_feedback_received",
            "total_evicted",
            "avg_quality_score",
            "low_quality_count",
        ]
        for field in required:
            assert field in stats


# ──────────────────────────────────────────────────────────────────
# 4. Cache Warmer (unit-level, no real cache needed)
# ──────────────────────────────────────────────────────────────────
from byte.processor.warmer import CacheWarmer


class MockCache:
    """Minimal cache mock for warmer tests."""

    def __init__(self) -> None:
        self.data_manager = type("DM", (), {"save": lambda self, q, a, e: None})()
        self.embedding_func = lambda x: [0.1, 0.2, 0.3]


class TestCacheWarmer:
    def test_warm_from_dict_counts(self) -> None:
        warmer = CacheWarmer(MockCache())
        result = warmer.warm_from_dict(
            [
                {"question": "What is AI?", "answer": "Artificial Intelligence"},
                {"question": "What is ML?", "answer": "Machine Learning"},
                {"question": "", "answer": "empty question"},  # should be skipped
            ]
        )
        assert result["seeded"] == 2
        assert result["skipped"] == 1
        assert result["total_processed"] == 3

    def test_warm_from_dict_empty(self) -> None:
        warmer = CacheWarmer(MockCache())
        result = warmer.warm_from_dict([])
        assert result["seeded"] == 0

    def test_warm_file_not_found(self) -> None:
        warmer = CacheWarmer(MockCache())
        with pytest.raises(FileNotFoundError):
            warmer.warm_from_file("/nonexistent/file.json")

    def test_warm_unsupported_format(self) -> None:
        warmer = CacheWarmer(MockCache())
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            f.write(b"<data/>")
            tmp = f.name
        try:
            with pytest.raises(ValueError):
                warmer.warm_from_file(tmp)
        finally:
            os.unlink(tmp)
