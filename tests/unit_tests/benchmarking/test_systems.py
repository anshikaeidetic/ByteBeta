from byte.benchmarking.systems import _reuse_confidence


def test_reuse_confidence_prefers_byte_trust_signal() -> None:
    confidence = _reuse_confidence(
        {
            "byte_trust": {"calibrated_confidence": 0.91},
            "byte_reasoning": {"confidence": 0.42},
        },
        "reuse",
    )

    assert confidence == 0.91
