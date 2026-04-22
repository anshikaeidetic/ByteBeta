from byte.processor.intent import extract_request_intent
from byte.processor.pre import canonicalize_text


def test_canonicalize_text_supports_braced_label_sets() -> None:
    content = canonicalize_text(
        'Ticket: "The app crashes whenever I click export."\n'
        "Classify this request and answer with exactly one label from {BILLING, TECHNICAL, SHIPPING}."
    )

    assert content.startswith("classify::ticket::billing|shipping|technical::")


def test_canonicalize_text_supports_quoted_extraction_keys_without_colon() -> None:
    content = canonicalize_text(
        'Return raw JSON only with keys "city" and "name".\nText: "Name: Alice. City: Paris."'
    )

    assert content.startswith("extract::json::city|name::")


def test_extract_request_intent_populates_slots_for_structured_workloads() -> None:
    classification = extract_request_intent(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        'Review: "I loved it."\n'
                        "Classify the sentiment and answer with exactly one label from {POSITIVE, NEGATIVE, NEUTRAL}."
                    ),
                }
            ]
        }
    )
    code_fix = extract_request_intent(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Fix the bug in this function.\n"
                        "Diagnostic: mutable default argument\n"
                        "```python\n"
                        "def add_item(item, items=[]):\n"
                        "    items.append(item)\n"
                        "    return items\n"
                        "```"
                    ),
                }
            ]
        }
    )

    assert classification.category == "classification"
    assert classification.slots["payload_key"] == "review"
    assert classification.slots["labels"] == "negative|neutral|positive"
    assert code_fix.category == "code_fix"
    assert code_fix.slots["language"] == "python"
    assert code_fix.slots["diagnostic"] == "mutable_default"
