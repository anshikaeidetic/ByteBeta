"""Public entry builders for optimization memory snapshots."""

from typing import Any

from byte.processor._optimization_summary import _success_rate


def _public_piece_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": entry.get("key"),
        "scope": entry.get("scope", ""),
        "piece_type": entry.get("piece_type", ""),
        "digest": entry.get("digest", ""),
        "preview": entry.get("preview", ""),
        "summary": entry.get("summary", ""),
        "chars": entry.get("chars", 0),
        "tokens_estimate": entry.get("tokens_estimate", 0),
        "compression": dict(entry.get("compression", {}) or {}),
        "metadata": dict(entry.get("metadata", {}) or {}),
        "source": entry.get("source", ""),
        "created_at": entry.get("created_at"),
        "updated_at": entry.get("updated_at"),
        "hits": entry.get("hits", 0),
    }


def _public_artifact_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": entry.get("key"),
        "scope": entry.get("scope", ""),
        "artifact_type": entry.get("artifact_type", ""),
        "fingerprint": entry.get("fingerprint", ""),
        "summary": entry.get("summary", ""),
        "sketch": entry.get("sketch", ""),
        "preview": entry.get("preview", ""),
        "chars": entry.get("chars", 0),
        "tokens_estimate": entry.get("tokens_estimate", 0),
        "compression": dict(entry.get("compression", {}) or {}),
        "metadata": dict(entry.get("metadata", {}) or {}),
        "source": entry.get("source", ""),
        "created_at": entry.get("created_at"),
        "updated_at": entry.get("updated_at"),
        "hits": entry.get("hits", 0),
    }


def _public_workflow_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": entry.get("key"),
        "scope": entry.get("scope", ""),
        "category": entry.get("category", ""),
        "route_key": entry.get("route_key", ""),
        "canonical_key": entry.get("canonical_key", ""),
        "tool_signature": entry.get("tool_signature", ""),
        "repo_fingerprint": entry.get("repo_fingerprint", ""),
        "artifact_fingerprint": entry.get("artifact_fingerprint", ""),
        "preferred_action": entry.get("preferred_action", ""),
        "route_preference": entry.get("route_preference", ""),
        "successes": entry.get("successes", 0),
        "failures": entry.get("failures", 0),
        "success_rate": round(_success_rate(entry), 4),
        "failed_actions": dict(entry.get("failed_actions", {}) or {}),
        "counterfactual_actions": dict(entry.get("counterfactual_actions", {}) or {}),
        "metadata": dict(entry.get("metadata", {}) or {}),
        "created_at": entry.get("created_at"),
        "updated_at": entry.get("updated_at"),
        "hits": entry.get("hits", 0),
    }


def _public_session_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": entry.get("key"),
        "scope": entry.get("scope", ""),
        "session_key": entry.get("session_key", ""),
        "artifact_type": entry.get("artifact_type", ""),
        "previous_digest": entry.get("previous_digest", ""),
        "current_digest": entry.get("current_digest", ""),
        "changed": bool(entry.get("changed", False)),
        "summary": entry.get("summary", ""),
        "metadata": dict(entry.get("metadata", {}) or {}),
        "created_at": entry.get("created_at"),
        "updated_at": entry.get("updated_at"),
        "hits": entry.get("hits", 0),
    }


__all__ = [name for name in globals() if not name.startswith("__")]


def _extract_request_intent(request_kwargs: dict[str, Any]) -> Any:
    from byte.processor.intent import extract_request_intent  # pylint: disable=C0415

    return extract_request_intent(request_kwargs)
