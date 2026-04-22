from byte.processor.fingerprint import selective_payload_fingerprint


def test_selective_payload_fingerprint_tracks_changes_beyond_preview_window() -> None:
    shared_prefix = "A" * 2100
    left = {
        "byte_retrieval_context": [
            {"title": "Shared", "snippet": shared_prefix + " invoice INV-9100"}
        ]
    }
    right = {
        "byte_retrieval_context": [
            {"title": "Shared", "snippet": shared_prefix + " invoice INV-9200"}
        ]
    }

    left_fp = selective_payload_fingerprint(left, ["byte_retrieval_context"])
    right_fp = selective_payload_fingerprint(right, ["byte_retrieval_context"])

    assert left_fp
    assert right_fp
    assert left_fp != right_fp
