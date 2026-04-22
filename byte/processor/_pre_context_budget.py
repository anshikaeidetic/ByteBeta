"""Context-note packing and budgeting helpers for request context compilation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from byte.processor._optimization_summary import compact_text, stable_digest
from byte.processor._pre_relevance import _lexical_overlap, _lexical_tokens
from byte.utils.multimodal import content_signature


def _append_compiled_context_notes(
    compiled: dict[str, Any],
    notes: list[str],
    *,
    prefix_messages: bool = False,
) -> None:
    combined = "\n\n".join(note for note in notes if note).strip()
    if not combined:
        return
    if compiled.get("messages"):
        if prefix_messages:
            messages = list(compiled["messages"])
            prefix_message = {
                "role": "system",
                "content": f"Byte compiled context:\n{combined}",
            }
            if not any(
                str(message.get("role", "") or "") == "system"
                and str(message.get("content", "") or "").strip() == prefix_message["content"]
                for message in messages
                if isinstance(message, dict)
            ):
                insert_at = (
                    1 if messages and str(messages[0].get("role", "") or "") == "system" else 0
                )
                messages.insert(insert_at, prefix_message)
                compiled["messages"] = messages
            return
        last_message = compiled["messages"][-1]
        content = last_message.get("content", "")
        if isinstance(content, list):
            updated = deepcopy(content)
            updated.append({"type": "text", "text": combined})
            last_message["content"] = updated
        else:
            last_message["content"] = f"{content}\n\n{combined}".strip()
        return
    if compiled.get("prompt") is not None:
        compiled["prompt"] = f"{compiled.get('prompt', '')}\n\n{combined}".strip()
        return
    if compiled.get("input") is not None:
        compiled["input"] = f"{compiled.get('input', '')}\n\n{combined}".strip()


def _compiled_primary_chars(compiled: dict[str, Any]) -> int:
    if compiled.get("messages"):
        return sum(
            len(
                content_signature(msg.get("content", ""))
                if isinstance(msg.get("content", ""), list)
                else str(msg.get("content", ""))
            )
            for msg in compiled["messages"]
        )
    if compiled.get("prompt") is not None:
        return len(str(compiled.get("prompt") or ""))
    if compiled.get("input") is not None:
        return len(str(compiled.get("input") or ""))
    return 0


def _fit_compiled_context_notes(
    notes: list[dict[str, Any]],
    *,
    max_chars: int,
    base_chars: int,
    total_aux_budget_ratio: float,
    cross_note_dedupe: bool,
) -> tuple[list[str], dict[str, int]]:
    if not notes:
        return [], {
            "cross_note_deduped_notes": 0,
            "aux_budget_pruned_notes": 0,
            "aux_budget_trimmed_chars": 0,
        }

    ratio = max(0.1, min(float(total_aux_budget_ratio or 0.65), 0.95))
    budget_cap = max(128, int(max_chars * ratio))
    remaining_total = max(0, int(max_chars or 0) - max(0, int(base_chars or 0)))
    if remaining_total <= 0:
        aux_budget = max(128, min(budget_cap, int(max_chars * 0.22)))
    else:
        aux_budget = max(128, min(budget_cap, remaining_total))

    selected: list[dict[str, Any]] = []
    selected_tokens: list[set] = []
    seen_digests = set()
    pruned = 0
    trimmed_chars = 0
    deduped = 0
    remaining = aux_budget

    ranked = sorted(
        notes,
        key=lambda item: (
            float(item.get("priority", 0.0) or 0.0),
            float(item.get("focus_score", 0.0) or 0.0),
            -int(item.get("order", 0) or 0),
        ),
        reverse=True,
    )

    for note_entry in ranked:
        note = str(note_entry.get("note") or "").strip()
        if not note:
            continue
        note_digest = str(note_entry.get("digest") or stable_digest(note))
        if note_digest in seen_digests:
            deduped += 1
            continue
        tokens = _lexical_tokens(note)
        if (
            cross_note_dedupe
            and tokens
            and any(_note_tokens_duplicate(tokens, prior) for prior in selected_tokens)
        ):
            deduped += 1
            continue
        separator = 0 if not selected else 2
        available = max(0, remaining - separator)
        if available <= 0:
            pruned += 1
            continue
        fitted = note
        if len(fitted) > available:
            min_chars = 72 if not selected else 96
            if available < min_chars:
                pruned += 1
                continue
            fitted = compact_text(fitted, max_chars=available)
            trimmed_chars += max(len(note) - len(fitted), 0)
        selected.append(
            {
                "order": int(note_entry.get("order", 0) or 0),
                "note": fitted,
            }
        )
        selected_tokens.append(tokens)
        seen_digests.add(note_digest)
        remaining -= len(fitted) + separator

    selected = sorted(selected, key=lambda item: item["order"])
    return [item["note"] for item in selected], {
        "cross_note_deduped_notes": deduped,
        "aux_budget_pruned_notes": pruned,
        "aux_budget_trimmed_chars": trimmed_chars,
    }


def _note_tokens_duplicate(left: set, right: set) -> bool:
    if not left or not right:
        return False
    overlap = _lexical_overlap(left, right)
    if overlap >= 0.74:
        return True
    return left.issubset(right) or right.issubset(left)


def _trim_text_middle(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep_each_side = max((max_chars - 32) // 2, 32)
    return f"{text[:keep_each_side]}\n...[byte context compiler trimmed middle]...\n{text[-keep_each_side:]}"
