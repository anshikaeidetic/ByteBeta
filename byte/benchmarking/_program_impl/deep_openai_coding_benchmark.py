import argparse
import concurrent.futures as cf
import json
import os
import random
import shutil
import statistics
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

from byte import Cache, Config  # pylint: disable=wrong-import-position
from byte._backends import openai as byte_openai  # pylint: disable=wrong-import-position
from byte.adapter.api import (  # pylint: disable=wrong-import-position
    init_cache,
    preview_model_route,
)
from byte.benchmarking._optional_runtime import create_openai_client
from byte.benchmarking._program_common import (
    call_with_retry as _common_call_with_retry,
)
from byte.benchmarking._program_common import (
    make_item as _common_make_item,
)
from byte.benchmarking._program_common import (
    normalized_answer as _common_normalized_answer,
)
from byte.benchmarking._program_common import (
    p95 as _common_p95,
)
from byte.benchmarking._program_common import (
    release_cache_tree as _common_release_cache_tree,
)
from byte.benchmarking._program_common import (
    usage_fields as _common_usage_fields,
)
from byte.benchmarking._program_defaults import load_program_defaults
from byte.processor.model_router import (
    clear_route_performance,  # pylint: disable=wrong-import-position
)
from byte.processor.pre import (  # pylint: disable=wrong-import-position
    last_content,
    normalized_last_content,
)
from byte.processor.shared_memory import (
    clear_shared_memory,  # pylint: disable=wrong-import-position
)

_DEFAULTS = load_program_defaults("deep_openai_coding_benchmark")

CHEAP_MODEL = _DEFAULTS.get("cheap_model", "gpt-4o-mini")
EXPENSIVE_MODEL = _DEFAULTS.get("expensive_model", "gpt-4o")
BYTE_MODES = list(_DEFAULTS.get("byte_modes", ["exact", "normalized", "hybrid"]))
BUG_LABELS = "MUTABLE_DEFAULT, OFF_BY_ONE, SYNTAX_ERROR, BROAD_EXCEPTION, NONE"
COMPLEXITY_LABELS = "O_1, O_N, O_N_SQUARED"
FRAMEWORK_LABELS = "PYTEST, UNITTEST, JEST"
PRICING = dict(_DEFAULTS.get("pricing", {}))
PRICING["sources"] = list(_DEFAULTS.get("pricing_sources", []))
PRICING["verified_on"] = _DEFAULTS.get("verified_on", "2026-03-10")


_make_item = _common_make_item


def _python_code_block(code: str) -> str:
    return f"```python\n{code.strip()}\n```"


def _bugfix_variants(file_path: str, alt_path: str, diagnostic: str, code: str) -> list[str]:
    code_block = _python_code_block(code)
    return [
        (
            "You are fixing a bug in the selected file.\n"
            f"File: {file_path}\n"
            f"Diagnostic: {diagnostic}\n"
            f"{code_block}\n"
            f"Return exactly one label from {{{BUG_LABELS}}}."
        ),
        (
            "You are debugging the selected code.\n"
            f"Path: {alt_path}\n"
            "Line 27\n"
            f"Error: {diagnostic}\n"
            f"{code_block}\n"
            f"Reply with exactly one label from {{{BUG_LABELS}}}."
        ),
    ]


def _explain_variants(file_path: str, alt_path: str, code: str) -> list[str]:
    code_block = _python_code_block(code)
    return [
        (
            "Explain this function in one sentence.\n"
            f"File: {file_path}\n"
            f"{code_block}\n"
            f"Return exactly one complexity label from {{{COMPLEXITY_LABELS}}}."
        ),
        (
            "Walk me through what this selected code does.\n"
            f"Path: {alt_path}\n"
            "Line 11\n"
            f"{code_block}\n"
            f"Answer with exactly one complexity label from {{{COMPLEXITY_LABELS}}}."
        ),
    ]


def _test_variants(file_path: str, alt_path: str, code: str) -> list[str]:
    code_block = _python_code_block(code)
    return [
        (
            "Write pytest tests for this helper.\n"
            f"File: {file_path}\n"
            f"{code_block}\n"
            f"Return exactly one framework label from {{{FRAMEWORK_LABELS}}}."
        ),
        (
            "Add unit tests using pytest for the selected function.\n"
            f"Path: {alt_path}\n"
            "Selection: function body\n"
            f"{code_block}\n"
            f"Reply with exactly one framework label from {{{FRAMEWORK_LABELS}}}."
        ),
    ]


def _docstring_variants(file_path: str, alt_path: str, code: str, token: str) -> list[str]:
    code_block = _python_code_block(code)
    return [
        (
            "Add a docstring for this function.\n"
            f"File: {file_path}\n"
            f"{code_block}\n"
            f"Return exactly {token} and nothing else."
        ),
        (
            "Write a documentation comment for the selected code.\n"
            f"Path: {alt_path}\n"
            f"{code_block}\n"
            f"Reply with exactly {token} and nothing else."
        ),
    ]


def _refactor_variants(file_path: str, alt_path: str, code: str, token: str) -> list[str]:
    code_block = _python_code_block(code)
    return [
        (
            "Refactor this function for readability.\n"
            f"File: {file_path}\n"
            f"{code_block}\n"
            f"Return exactly {token} and nothing else."
        ),
        (
            "Clean up the selected code and improve readability.\n"
            f"Path: {alt_path}\n"
            "Line 18\n"
            f"{code_block}\n"
            f"Reply with exactly {token} and nothing else."
        ),
    ]


def _build_sequential_scenarios() -> list[dict[str, Any]]:
    rng = random.Random(23)

    bugfix_groups = [
        (
            "cursor_bug_mutable_default",
            "MUTABLE_DEFAULT",
            "mutable default argument",
            "src/cart.py:14",
            "app/cart.py",
            "def add_item(item, items=[]):\n    items.append(item)\n    return items",
        ),
        (
            "cursor_bug_off_by_one",
            "OFF_BY_ONE",
            "off by one loop bound",
            "src/metrics.py:33",
            "app/metrics.py",
            "def sum_all(values):\n    total = 0\n    for i in range(len(values) + 1):\n        total += values[i]\n    return total",
        ),
        (
            "cursor_bug_syntax_error",
            "SYNTAX_ERROR",
            "invalid syntax",
            "src/parser.py:22",
            "app/parser.py",
            "def parse(value)\n    return value.strip()",
        ),
        (
            "cursor_bug_broad_exception",
            "BROAD_EXCEPTION",
            "broad exception clause",
            "src/loader.py:51",
            "app/loader.py",
            "def parse_int(value):\n    try:\n        return int(value)\n    except:\n        return 0",
        ),
    ]
    bugfix_items: list[dict[str, Any]] = []
    for group, expected, diagnostic, file_path, alt_path, code in bugfix_groups:
        for variant_index, prompt in enumerate(
            _bugfix_variants(file_path, alt_path, diagnostic, code), 1
        ):
            bugfix_items.append(
                _make_item(prompt, expected, group, f"v{variant_index}", "code_fix")
            )

    explain_groups = [
        (
            "cursor_explain_linear",
            "O_N",
            "src/math_utils.py",
            "app/math_utils.py",
            "def total(values):\n    result = 0\n    for value in values:\n        result += value\n    return result",
        ),
        (
            "cursor_explain_nested",
            "O_N_SQUARED",
            "src/matrix.py",
            "app/matrix.py",
            "def pair_sum(matrix):\n    total = 0\n    for row in matrix:\n        for value in row:\n            total += value\n    return total",
        ),
        (
            "cursor_explain_lookup",
            "O_1",
            "src/config.py",
            "app/config.py",
            "def get_timeout(config):\n    return config['timeout']",
        ),
    ]
    explain_items: list[dict[str, Any]] = []
    for group, expected, file_path, alt_path, code in explain_groups:
        for variant_index, prompt in enumerate(_explain_variants(file_path, alt_path, code), 1):
            explain_items.append(
                _make_item(prompt, expected, group, f"v{variant_index}", "code_explanation")
            )

    test_groups = [
        (
            "cursor_tests_slugify",
            "PYTEST",
            "src/helpers.py",
            "app/helpers.py",
            "def slugify(value):\n    return value.strip().lower().replace(' ', '-')",
        ),
        (
            "cursor_tests_parse_int",
            "PYTEST",
            "src/parse.py",
            "app/parse.py",
            "def parse_int(value):\n    if value is None:\n        return 0\n    return int(value)",
        ),
        (
            "cursor_tests_is_even",
            "PYTEST",
            "src/number.py",
            "app/number.py",
            "def is_even(value):\n    return value % 2 == 0",
        ),
    ]
    test_items: list[dict[str, Any]] = []
    for group, expected, file_path, alt_path, code in test_groups:
        for variant_index, prompt in enumerate(_test_variants(file_path, alt_path, code), 1):
            test_items.append(
                _make_item(prompt, expected, group, f"v{variant_index}", "test_generation")
            )

    prewarmed_seed = [
        {
            "question": _bugfix_variants(
                "src/cart.py:14",
                "app/cart.py",
                "mutable default argument",
                "def add_item(item, items=[]):\n    items.append(item)\n    return items",
            )[0],
            "answer": "MUTABLE_DEFAULT",
            "model": CHEAP_MODEL,
        },
        {
            "question": _explain_variants(
                "src/math_utils.py",
                "app/math_utils.py",
                "def total(values):\n    result = 0\n    for value in values:\n        result += value\n    return result",
            )[0],
            "answer": "O_N",
            "model": CHEAP_MODEL,
        },
    ]
    prewarmed_items = [
        _make_item(
            _bugfix_variants(
                "src/cart.py:14",
                "app/cart.py",
                "mutable default argument",
                "def add_item(item, items=[]):\n    items.append(item)\n    return items",
            )[1],
            "MUTABLE_DEFAULT",
            "warm_bugfix_mutable_default",
            "v2",
            "code_fix",
        ),
        _make_item(
            _explain_variants(
                "src/math_utils.py",
                "app/math_utils.py",
                "def total(values):\n    result = 0\n    for value in values:\n        result += value\n    return result",
            )[1],
            "O_N",
            "warm_code_explain_linear",
            "v2",
            "code_explanation",
        ),
        _make_item(
            _test_variants(
                "src/helpers.py",
                "app/helpers.py",
                "def slugify(value):\n    return value.strip().lower().replace(' ', '-')",
            )[0],
            "PYTEST",
            "warm_test_slugify",
            "v1",
            "test_generation",
        ),
        _make_item(
            _docstring_variants(
                "src/helpers.py",
                "app/helpers.py",
                "def slugify(value):\n    return value.strip().lower().replace(' ', '-')",
                "DOCSTRING_READY",
            )[0],
            "DOCSTRING_READY",
            "warm_docstring_slugify",
            "v1",
            "documentation",
        ),
    ]

    mixed_items: list[dict[str, Any]] = []
    mixed_items.extend(bugfix_items[:4])
    mixed_items.extend(explain_items[:4])
    mixed_items.extend(test_items[:4])
    mixed_items.extend(
        [
            _make_item(
                _docstring_variants(
                    "src/cleanup.py",
                    "app/cleanup.py",
                    "def normalize_name(value):\n    return value.strip().title()",
                    "DOCSTRING_READY",
                )[0],
                "DOCSTRING_READY",
                "mixed_docstring_cleanup",
                "v1",
                "documentation",
            ),
            _make_item(
                _docstring_variants(
                    "src/cleanup.py",
                    "app/cleanup.py",
                    "def normalize_name(value):\n    return value.strip().title()",
                    "DOCSTRING_READY",
                )[1],
                "DOCSTRING_READY",
                "mixed_docstring_cleanup",
                "v2",
                "documentation",
            ),
            _make_item(
                _refactor_variants(
                    "src/orders.py",
                    "app/orders.py",
                    "def build_message(name, total):\n    return 'Customer ' + name + ' owes ' + str(total)",
                    "READABILITY_REFACTOR",
                )[0],
                "READABILITY_REFACTOR",
                "mixed_refactor_orders",
                "v1",
                "code_refactor",
            ),
            _make_item(
                _refactor_variants(
                    "src/orders.py",
                    "app/orders.py",
                    "def build_message(name, total):\n    return 'Customer ' + name + ' owes ' + str(total)",
                    "READABILITY_REFACTOR",
                )[1],
                "READABILITY_REFACTOR",
                "mixed_refactor_orders",
                "v2",
                "code_refactor",
            ),
            _make_item(
                "Explain this function in one sentence.\n"
                "File: src/unique_a.py\n"
                "```python\n"
                "def first(values):\n    return values[0]\n"
                "```\n"
                f"Return exactly one complexity label from {{{COMPLEXITY_LABELS}}}.",
                "O_1",
                "mixed_unique_explain_a",
                "base",
                "code_explanation",
            ),
            _make_item(
                "Explain this function in one sentence.\n"
                "File: src/unique_b.py\n"
                "```python\n"
                "def contains_zero(values):\n    for value in values:\n        if value == 0:\n            return True\n    return False\n"
                "```\n"
                f"Return exactly one complexity label from {{{COMPLEXITY_LABELS}}}.",
                "O_N",
                "mixed_unique_explain_b",
                "base",
                "code_explanation",
            ),
            _make_item(
                "Write pytest tests for this helper.\n"
                "File: src/unique_c.py\n"
                "```python\n"
                "def clamp(value, lower, upper):\n    return max(lower, min(value, upper))\n"
                "```\n"
                f"Return exactly one framework label from {{{FRAMEWORK_LABELS}}}.",
                "PYTEST",
                "mixed_unique_test_c",
                "base",
                "test_generation",
            ),
            _make_item(
                "Fix the bug in this Python function.\n"
                "File: src/unique_d.py\n"
                "Diagnostic: broad exception clause\n"
                "```python\n"
                "def load_port(value):\n    try:\n        return int(value)\n    except:\n        return 8080\n"
                "```\n"
                f"Return exactly one label from {{{BUG_LABELS}}}.",
                "BROAD_EXCEPTION",
                "mixed_unique_bug_d",
                "base",
                "code_fix",
            ),
        ]
    )
    rng.shuffle(mixed_items)

    return [
        {
            "name": "cursor_bugfix_templates_8",
            "description": "Bug-fix prompts with file paths, diagnostics, and selected-code blocks. Variants change wrapper wording, file paths, and line numbers while keeping the same code bug.",
            "items": bugfix_items,
        },
        {
            "name": "cursor_explain_templates_6",
            "description": "Selected-code explanation prompts that ask for complexity labels. This approximates repeated 'explain this code' editor traffic.",
            "items": explain_items,
        },
        {
            "name": "cursor_test_templates_6",
            "description": "Test-generation prompt shells for the same selected function, phrased the way editor assistants usually ask for pytest coverage.",
            "items": test_items,
        },
        {
            "name": "cursor_prewarmed_hotset_4",
            "description": "Hot coding prompts pre-seeded before live traffic so the first editor user avoids the cold miss.",
            "items": prewarmed_items,
            "warm_data": prewarmed_seed,
        },
        {
            "name": "cursor_mixed_workload_16",
            "description": "A practical editor-assistant mix of bug fixes, code explanations, test-generation asks, docstrings, refactors, and true one-off code requests.",
            "items": mixed_items,
        },
    ]


def _build_concurrent_scenario() -> dict[str, Any]:
    items = [
        _make_item(
            _bugfix_variants(
                "src/cart.py:14",
                "app/cart.py",
                "mutable default argument",
                "def add_item(item, items=[]):\n    items.append(item)\n    return items",
            )[0],
            "MUTABLE_DEFAULT",
            "burst_bugfix_a",
            "v1",
            "code_fix",
        ),
        _make_item(
            _bugfix_variants(
                "src/cart.py:14",
                "app/cart.py",
                "mutable default argument",
                "def add_item(item, items=[]):\n    items.append(item)\n    return items",
            )[1],
            "MUTABLE_DEFAULT",
            "burst_bugfix_a",
            "v2",
            "code_fix",
        ),
        _make_item(
            _bugfix_variants(
                "src/metrics.py:33",
                "app/metrics.py",
                "off by one loop bound",
                "def sum_all(values):\n    total = 0\n    for i in range(len(values) + 1):\n        total += values[i]\n    return total",
            )[0],
            "OFF_BY_ONE",
            "burst_bugfix_b",
            "v1",
            "code_fix",
        ),
        _make_item(
            _bugfix_variants(
                "src/metrics.py:33",
                "app/metrics.py",
                "off by one loop bound",
                "def sum_all(values):\n    total = 0\n    for i in range(len(values) + 1):\n        total += values[i]\n    return total",
            )[1],
            "OFF_BY_ONE",
            "burst_bugfix_b",
            "v2",
            "code_fix",
        ),
    ]
    items.extend(
        [
            _make_item(
                _explain_variants(
                    "src/math_utils.py",
                    "app/math_utils.py",
                    "def total(values):\n    result = 0\n    for value in values:\n        result += value\n    return result",
                )[0],
                "O_N",
                "burst_explain_a",
                "v1",
                "code_explanation",
            ),
            _make_item(
                _explain_variants(
                    "src/math_utils.py",
                    "app/math_utils.py",
                    "def total(values):\n    result = 0\n    for value in values:\n        result += value\n    return result",
                )[1],
                "O_N",
                "burst_explain_a",
                "v2",
                "code_explanation",
            ),
            _make_item(
                _test_variants(
                    "src/helpers.py",
                    "app/helpers.py",
                    "def slugify(value):\n    return value.strip().lower().replace(' ', '-')",
                )[0],
                "PYTEST",
                "burst_tests_a",
                "v1",
                "test_generation",
            ),
            _make_item(
                _test_variants(
                    "src/helpers.py",
                    "app/helpers.py",
                    "def slugify(value):\n    return value.strip().lower().replace(' ', '-')",
                )[1],
                "PYTEST",
                "burst_tests_a",
                "v2",
                "test_generation",
            ),
            _make_item(
                _docstring_variants(
                    "src/cleanup.py",
                    "app/cleanup.py",
                    "def normalize_name(value):\n    return value.strip().title()",
                    "DOCSTRING_READY",
                )[0],
                "DOCSTRING_READY",
                "burst_doc_a",
                "v1",
                "documentation",
            ),
            _make_item(
                _docstring_variants(
                    "src/cleanup.py",
                    "app/cleanup.py",
                    "def normalize_name(value):\n    return value.strip().title()",
                    "DOCSTRING_READY",
                )[1],
                "DOCSTRING_READY",
                "burst_doc_a",
                "v2",
                "documentation",
            ),
            _make_item(
                "Explain this function in one sentence.\n"
                "File: src/unique_a.py\n"
                "```python\n"
                "def first(values):\n    return values[0]\n"
                "```\n"
                f"Return exactly one complexity label from {{{COMPLEXITY_LABELS}}}.",
                "O_1",
                "burst_unique_a",
                "base",
                "code_explanation",
            ),
            _make_item(
                "Write pytest tests for this helper.\n"
                "File: src/unique_b.py\n"
                "```python\n"
                "def clamp(value, lower, upper):\n    return max(lower, min(value, upper))\n"
                "```\n"
                f"Return exactly one framework label from {{{FRAMEWORK_LABELS}}}.",
                "PYTEST",
                "burst_unique_b",
                "base",
                "test_generation",
            ),
        ]
    )
    return {
        "name": "cursor_concurrent_burst_12",
        "description": "Concurrent editor-like traffic with repeated bug-fix, explain, test, and docstring requests landing at the same time.",
        "items": items,
    }


def _build_routing_scenario() -> dict[str, Any]:
    route_fix_code = _python_code_block(
        "def add_item(item, items=[]):\n    items.append(item)\n    return items"
    )
    route_refactor_code = _python_code_block(
        "def build_message(name, total):\n    return 'Customer ' + name + ' owes ' + str(total)"
    )
    route_test_code = _python_code_block(
        "def parse_int(value):\n    if value is None:\n        return 0\n    return int(value)"
    )
    route_debug_code = _python_code_block(
        "def parse_int(value):\n    try:\n        return int(value)\n    except:\n        return 0"
    )
    items = [
        _make_item(
            _explain_variants(
                "src/math_utils.py",
                "app/math_utils.py",
                "def total(values):\n    result = 0\n    for value in values:\n        result += value\n    return result",
            )[0],
            "O_N",
            "route_explain_a",
            "v1",
            "code_explanation",
        ),
        _make_item(
            _explain_variants(
                "src/config.py",
                "app/config.py",
                "def get_timeout(config):\n    return config['timeout']",
            )[0],
            "O_1",
            "route_explain_b",
            "v1",
            "code_explanation",
        ),
        _make_item(
            _docstring_variants(
                "src/helpers.py",
                "app/helpers.py",
                "def slugify(value):\n    return value.strip().lower().replace(' ', '-')",
                "DOCSTRING_READY",
            )[0],
            "DOCSTRING_READY",
            "route_doc_a",
            "v1",
            "documentation",
        ),
        _make_item(
            _docstring_variants(
                "src/cleanup.py",
                "app/cleanup.py",
                "def normalize_name(value):\n    return value.strip().title()",
                "DOCSTRING_DONE",
            )[0],
            "DOCSTRING_DONE",
            "route_doc_b",
            "v1",
            "documentation",
        ),
        _make_item(
            (
                "Fix the production bug in this function. Think through the failure mode, compare likely fixes, and then return exactly BYTE_CODE_FIX_A and nothing else.\n"
                "File: src/cart.py:14\n"
                "Diagnostic: mutable default argument causing shared state in the checkout flow.\n"
                + route_fix_code
            ),
            "BYTE_CODE_FIX_A",
            "route_fix_a",
            "v1",
            "code_fix",
            max_tokens=16,
        ),
        _make_item(
            (
                "Refactor this function for readability, compare tradeoffs, preserve behavior, and after your analysis reply with exactly BYTE_CODE_REFACTOR_B and nothing else.\n"
                "File: src/orders.py:48\n" + route_refactor_code
            ),
            "BYTE_CODE_REFACTOR_B",
            "route_refactor_b",
            "v1",
            "code_refactor",
            max_tokens=16,
        ),
        _make_item(
            (
                "Write a full pytest suite with happy path, edge cases, and failure coverage for this function, reason about the branch coverage first, and then return exactly BYTE_CODE_TEST_C and nothing else.\n"
                "File: src/parse.py:19\n" + route_test_code
            ),
            "BYTE_CODE_TEST_C",
            "route_tests_c",
            "v1",
            "test_generation",
            max_tokens=16,
        ),
        _make_item(
            (
                "Debug the failing request path, analyze the error handling, compare the rollback options, and after that reply with exactly BYTE_CODE_DEBUG_D and nothing else.\n"
                "File: src/loader.py:51\n"
                "Diagnostic: broad exception clause hides parser failures in production.\n"
                + route_debug_code
            ),
            "BYTE_CODE_DEBUG_D",
            "route_debug_d",
            "v1",
            "code_fix",
            max_tokens=16,
        ),
    ]
    return {
        "name": "cursor_routing_blend_8",
        "description": "Cursor-style mix of short explanations and docstrings versus heavier fix, test, refactor, and debug requests so Byte can route cheap work to gpt-4o-mini and keep hard work on gpt-4o.",
        "items": items,
    }


_usage_fields = _common_usage_fields


def _pricing_key(model_name: str) -> str:
    normalized = (model_name or "").lower()
    if normalized.startswith("gpt-4o-mini"):
        return "gpt-4o-mini"
    return "gpt-4o"


def _pricing_cost(model_name: str, usage: Any) -> float:
    fields = _usage_fields(usage)
    price = PRICING[_pricing_key(model_name)]
    prompt_tokens = fields["prompt_tokens"]
    cached_prompt_tokens = min(fields["cached_prompt_tokens"], prompt_tokens)
    completion_tokens = fields["completion_tokens"]
    uncached_prompt_tokens = max(prompt_tokens - cached_prompt_tokens, 0)
    return (
        (uncached_prompt_tokens / 1_000_000) * price["input"]
        + (cached_prompt_tokens / 1_000_000) * price["cached_input"]
        + (completion_tokens / 1_000_000) * price["output"]
    )


_normalized_answer = _common_normalized_answer


def _response_record(
    *,
    status_code: int,
    latency_ms: float,
    byte_flag: bool,
    model_name: str,
    route_info: dict[str, Any] | None,
    text: str | None,
    usage: Any,
    item: dict[str, Any],
    error: str = "",
) -> dict[str, Any]:
    fields = _usage_fields(usage)
    return {
        "status_code": status_code,
        "latency_ms": round(latency_ms, 2),
        "byte": byte_flag,
        "prompt_tokens": fields["prompt_tokens"],
        "cached_prompt_tokens": fields["cached_prompt_tokens"],
        "completion_tokens": fields["completion_tokens"],
        "cost_usd": round(_pricing_cost(model_name, usage), 8),
        "text": text,
        "expected": item["expected"],
        "group": item["group"],
        "variant": item["variant"],
        "kind": item["kind"],
        "model": model_name,
        "route_info": route_info or {},
        "error": error,
        "correct": _normalized_answer(text) == _normalized_answer(item["expected"]),
    }


_call_with_retry = _common_call_with_retry


def _direct_request(api_key: str, item: dict[str, Any], model: str) -> dict[str, Any]:
    client = create_openai_client(api_key=api_key)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": item["prompt"]}],
        "temperature": 0,
        "max_tokens": int(item.get("max_tokens", 12) or 12),
    }
    start = time.perf_counter()
    try:
        response = _call_with_retry(lambda: client.chat.completions.create(**payload))
        latency_ms = (time.perf_counter() - start) * 1000
        text = response.choices[0].message.content or ""
        model_name = str(getattr(response, "model", "") or model)
        return _response_record(
            status_code=200,
            latency_ms=latency_ms,
            byte_flag=False,
            model_name=model_name,
            route_info=None,
            text=text,
            usage=response.usage,
            item=item,
        )
    except Exception as exc:  # pylint: disable=broad-except
        latency_ms = (time.perf_counter() - start) * 1000
        return _response_record(
            status_code=599,
            latency_ms=latency_ms,
            byte_flag=False,
            model_name=model,
            route_info=None,
            text=None,
            usage=None,
            item=item,
            error=str(exc),
        )


def _base_config(scope: str, *, routed: bool) -> Config:
    return Config(
        enable_token_counter=False,
        embedding_cache_size=20000,
        tiered_cache=True,
        tier1_max_size=2048,
        tier1_promote_on_write=True,
        async_write_back=True,
        memory_scope=scope,
        intent_memory=True,
        model_namespace=True,
        tool_namespace=True,
        context_fingerprint=True,
        routing_long_prompt_chars=900,
        routing_multi_turn_threshold=4,
        routing_verify_cheap_responses=True,
        routing_verify_min_score=0.8,
        routing_adaptive=True,
        routing_adaptive_min_samples=3,
        routing_adaptive_quality_floor=0.75,
        cache_admission_min_score=0.2,
        model_routing=routed,
        routing_cheap_model=CHEAP_MODEL if routed else None,
        routing_expensive_model=EXPENSIVE_MODEL if routed else None,
        routing_default_model=EXPENSIVE_MODEL if routed else None,
        semantic_allowed_categories=[
            "question_answer",
            "summarization",
            "comparison",
            "instruction",
        ],
    )


def _configure_cache(
    cache_obj: Cache, cache_dir: str, mode: str, *, scope: str, routed: bool, warm_data=None
) -> None:
    config = _base_config(scope, routed=routed)
    init_cache(
        mode=mode,
        data_dir=cache_dir,
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=config,
        exact_config=config,
        normalized_config=config,
        semantic_config=config,
        warm_data=warm_data,
    )


def _memory_capture(cache_obj: Cache) -> dict[str, Any]:
    snapshot = cache_obj.export_memory_snapshot(tool_result_limit=8)
    route_tiers = Counter()
    for entry in snapshot.get("ai_memory", {}).get("entries", []):
        tier = ((entry.get("metadata") or {}).get("model_route") or {}).get("tier")
        if tier:
            route_tiers[tier] += 1
    return {
        "summary": cache_obj.memory_summary(),
        "snapshot_stats": {
            "intent_records": snapshot.get("intent_graph", {}).get("total_records", 0),
            "ai_entries": snapshot.get("ai_memory", {}).get("stats", {}).get("total_entries", 0),
            "route_tiers": dict(route_tiers),
        },
        "recent_interactions": cache_obj.recent_interactions(limit=5),
    }


_release_cache_tree = _common_release_cache_tree


def _byte_request(
    api_key: str,
    cache_obj: Cache,
    item: dict[str, Any],
    *,
    model: str,
    scenario_name: str,
) -> dict[str, Any]:
    request_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": item["prompt"]}],
        "temperature": 0,
        "max_tokens": int(item.get("max_tokens", 12) or 12),
    }
    route_info = preview_model_route(dict(request_kwargs), cache_obj=cache_obj)
    payload = dict(request_kwargs)
    payload.update(
        {
            "api_key": api_key,
            "cache_obj": cache_obj,
            "byte_memory": {
                "provider": "openai",
                "metadata": {
                    "scenario": scenario_name,
                    "group": item["group"],
                    "variant": item["variant"],
                    "kind": item["kind"],
                },
            },
        }
    )
    start = time.perf_counter()
    try:
        response = _call_with_retry(lambda: byte_openai.ChatCompletion.create(**payload))
        latency_ms = (time.perf_counter() - start) * 1000
        text = (((response or {}).get("choices") or [{}])[0].get("message") or {}).get("content")
        model_name = str(
            (response or {}).get("model") or (route_info or {}).get("selected_model") or model
        )
        return _response_record(
            status_code=200,
            latency_ms=latency_ms,
            byte_flag=bool((response or {}).get("byte")),
            model_name=model_name,
            route_info=route_info,
            text=text,
            usage=(response or {}).get("usage"),
            item=item,
        )
    except Exception as exc:  # pylint: disable=broad-except
        latency_ms = (time.perf_counter() - start) * 1000
        fallback_model = (route_info or {}).get("selected_model") or model
        return _response_record(
            status_code=599,
            latency_ms=latency_ms,
            byte_flag=False,
            model_name=str(fallback_model),
            route_info=route_info,
            text=None,
            usage=None,
            item=item,
            error=str(exc),
        )


_p95 = _common_p95


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [record["latency_ms"] for record in records]
    cached = sum(1 for record in records if record["byte"])
    correct = sum(1 for record in records if record["correct"])
    model_counts = Counter(record["model"] for record in records if record.get("model"))
    route_tiers = Counter(
        (record.get("route_info") or {}).get("tier")
        for record in records
        if (record.get("route_info") or {}).get("tier")
    )
    return {
        "request_count": len(records),
        "cached_count": cached,
        "miss_count": len(records) - cached,
        "error_count": sum(1 for record in records if record["status_code"] != 200),
        "hit_ratio": round(cached / len(records), 4) if records else 0.0,
        "accuracy_ratio": round(correct / len(records), 4) if records else 0.0,
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": _p95(latencies),
        "total_prompt_tokens": sum(record["prompt_tokens"] for record in records),
        "total_cached_prompt_tokens": sum(record["cached_prompt_tokens"] for record in records),
        "total_completion_tokens": sum(record["completion_tokens"] for record in records),
        "total_cost_usd": round(sum(record["cost_usd"] for record in records), 8),
        "models": dict(model_counts),
        "route_tiers": dict(route_tiers),
        "sample": records[:5],
    }


def _run_direct_sequence(
    api_key: str, items: list[dict[str, Any]], *, model: str
) -> dict[str, Any]:
    return _summarize_records([_direct_request(api_key, item, model) for item in items])


def _run_byte_sequence(
    api_key: str,
    items: list[dict[str, Any]],
    *,
    mode: str,
    model: str,
    routed: bool,
    scenario_name: str,
    warm_data=None,
    capture_memory: bool = False,
) -> dict[str, Any]:
    cache_dir = tempfile.mkdtemp(prefix=f"coding-{mode}-")
    scope = f"coding::{mode}::{scenario_name}::{int(time.time() * 1000)}"
    clear_shared_memory(scope)
    cache_obj = Cache()
    try:
        clear_route_performance()
        _configure_cache(
            cache_obj, cache_dir, mode, scope=scope, routed=routed, warm_data=warm_data
        )
        records = [
            _byte_request(
                api_key,
                cache_obj,
                item,
                model=model,
                scenario_name=scenario_name,
            )
            for item in items
        ]
        result = _summarize_records(records)
        result["prewarmed"] = bool(warm_data)
        if capture_memory:
            result["memory"] = _memory_capture(cache_obj)
        return result
    finally:
        _release_cache_tree(cache_obj)
        shutil.rmtree(cache_dir, ignore_errors=True)
        clear_shared_memory(scope)


def _run_direct_concurrent(
    api_key: str, items: list[dict[str, Any]], *, model: str
) -> dict[str, Any]:
    wall_start = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=len(items)) as pool:
        records = list(pool.map(lambda item: _direct_request(api_key, item, model), items))
    result = _summarize_records(records)
    result["wall_time_ms"] = round((time.perf_counter() - wall_start) * 1000, 2)
    return result


def _run_byte_concurrent(
    api_key: str,
    items: list[dict[str, Any]],
    *,
    mode: str,
    model: str,
    scenario_name: str,
) -> dict[str, Any]:
    cache_dir = tempfile.mkdtemp(prefix=f"coding-burst-{mode}-")
    scope = f"coding::{mode}::{scenario_name}::burst::{int(time.time() * 1000)}"
    clear_shared_memory(scope)
    cache_obj = Cache()
    try:
        clear_route_performance()
        _configure_cache(cache_obj, cache_dir, mode, scope=scope, routed=False, warm_data=None)
        wall_start = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=len(items)) as pool:
            records = list(
                pool.map(
                    lambda item: _byte_request(
                        api_key,
                        cache_obj,
                        item,
                        model=model,
                        scenario_name=scenario_name,
                    ),
                    items,
                )
            )
        result = _summarize_records(records)
        result["wall_time_ms"] = round((time.perf_counter() - wall_start) * 1000, 2)
        return result
    finally:
        _release_cache_tree(cache_obj)
        shutil.rmtree(cache_dir, ignore_errors=True)
        clear_shared_memory(scope)


def _scenario_summary(
    name: str,
    description: str,
    items: list[dict[str, Any]],
    runs: dict[str, dict[str, Any]],
    *,
    baseline_key: str = "direct",
    warm_data=None,
) -> dict[str, Any]:
    summary = {
        "name": name,
        "description": description,
        "request_count": len(items),
        "unique_prompt_count": len({item["prompt"] for item in items}),
        "logical_group_count": len({item["group"] for item in items}),
        "prewarmed_seed_count": len(warm_data or []),
        "runs": runs,
    }
    baseline_cost = runs[baseline_key]["total_cost_usd"]
    baseline_latency = runs[baseline_key]["avg_latency_ms"]
    for key, data in runs.items():
        if key == baseline_key:
            data["saved_vs_baseline_usd"] = 0.0
            data["savings_ratio"] = 0.0
            data["latency_delta_ms"] = 0.0
            continue
        data["saved_vs_baseline_usd"] = round(baseline_cost - data["total_cost_usd"], 8)
        data["savings_ratio"] = (
            round((baseline_cost - data["total_cost_usd"]) / baseline_cost, 4)
            if baseline_cost
            else 0.0
        )
        data["latency_delta_ms"] = round(data["avg_latency_ms"] - baseline_latency, 2)
    return summary


def _render_money(value: float) -> str:
    return f"${value:.8f}"


def _render_mode_line(name: str, data: dict[str, Any], *, baseline_label: str = "direct") -> str:
    flags = []
    if data.get("accuracy_ratio", 1.0) < 1.0:
        flags.append("accuracy<1.0")
    if data.get("prewarmed"):
        flags.append("prewarmed=true")
    if data.get("error_count"):
        flags.append(f"errors={data['error_count']}")
    models = ", ".join(
        f"{model}:{count}" for model, count in sorted((data.get("models") or {}).items())
    )
    routes = ", ".join(
        f"{tier}:{count}" for tier, count in sorted((data.get("route_tiers") or {}).items())
    )
    suffix = f" [{'; '.join(flags)}]" if flags else ""
    model_suffix = f", models=({models})" if models else ""
    route_suffix = f", route_tiers=({routes})" if routes else ""
    return (
        f"- {name}: cost={_render_money(data['total_cost_usd'])}, hit_ratio={data['hit_ratio']}, "
        f"accuracy={data['accuracy_ratio']}, avg_latency={data['avg_latency_ms']} ms, "
        f"p95_latency={data['p95_latency_ms']} ms, saved_vs_{baseline_label}={_render_money(data.get('saved_vs_baseline_usd', 0.0))}, "
        f"savings_ratio={data.get('savings_ratio', 0.0)}{model_suffix}{route_suffix}{suffix}"
    )


def _render_memory_block(memory: dict[str, Any]) -> list[str]:
    summary = memory.get("summary", {})
    snapshot_stats = memory.get("snapshot_stats", {})
    lines = [
        f"- Intent records={snapshot_stats.get('intent_records', 0)}, ai_entries={snapshot_stats.get('ai_entries', 0)}, route_tiers={snapshot_stats.get('route_tiers', {})}",
        f"- Intent graph summary={summary.get('intent_graph', {})}",
        f"- AI memory summary={summary.get('ai_memory', {})}",
    ]
    recent = memory.get("recent_interactions", [])[:3]
    if recent:
        previews = [
            {
                "category": entry.get("category"),
                "model": entry.get("model"),
                "source": entry.get("last_source"),
                "hits": entry.get("hits"),
                "question_digest": entry.get("question_digest"),
            }
            for entry in recent
        ]
        lines.append(f"- Recent interactions={previews}")
    return lines


def _render_report(results: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# ByteAI Cache OpenAI Coding Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {results['generated_at']}")
    lines.append(f"Benchmark model: `{results['benchmark_model']}`")
    lines.append(f"Routing baseline model: `{results['expensive_model']}`")
    lines.append("")
    lines.append(
        "This benchmark simulates Cursor-style editor traffic using file paths, line numbers, diagnostics, selected-code blocks, and repeated coding prompt templates."
    )
    lines.append(
        "The implementation changes exercised here live in ByteAI Cache's shared cache/routing layer, so they apply across all adapters even though the live run below uses OpenAI."
    )
    lines.append("")
    lines.append("## Features Exercised")
    lines.append("")
    lines.append(
        "- Code-aware prompt canonicalizers for bug-fix, explain-code, test-generation, docstring, and refactor request shells."
    )
    lines.append(
        "- Path and line-number normalization so equivalent editor requests can reuse cache entries safely."
    )
    lines.append(
        "- Hybrid cache path tuned for coding workloads where normalized reuse is safe but broad semantic reuse is too risky."
    )
    lines.append(
        "- Smart model routing that keeps short explain/doc tasks cheap while sending fix/refactor/test/debug asks to a stronger model."
    )
    lines.append("- Provider-agnostic AI memory capture for routed coding interactions.")
    lines.append("")
    lines.append("## Pricing Source")
    lines.append("")
    lines.append(f"Verified on {results['pricing']['verified_on']}:")
    for url in results["pricing"]["sources"]:
        lines.append(f"- {url}")
    lines.append("")
    lines.append("## Why This Approximates Cursor")
    lines.append("")
    lines.append(
        "- Requests include selected-code blocks, file paths, line numbers, and diagnostics rather than generic chat questions."
    )
    lines.append(
        "- Repeated workloads use the same code with different wrapper instructions, matching how editor agents often vary prompt shells around the same task."
    )
    lines.append(
        "- Routing compares a 'send everything to the strong model' baseline against ByteAI Cache choosing between cheap and strong models on a coding mix."
    )
    lines.append("")
    lines.append("## Sequential Cache Results")
    lines.append("")
    for scenario in results["sequential_scenarios"]:
        lines.append(f"### {scenario['name']}")
        lines.append(f"- {scenario['description']}")
        lines.append(
            f"- Requests={scenario['request_count']}, unique_prompts={scenario['unique_prompt_count']}, "
            f"logical_groups={scenario['logical_group_count']}, prewarmed_seed_count={scenario['prewarmed_seed_count']}"
        )
        lines.append(_render_mode_line("Direct", scenario["runs"]["direct"]))
        for mode in BYTE_MODES:
            lines.append(_render_mode_line(f"ByteAI Cache {mode}", scenario["runs"][mode]))
        lines.append("")
    lines.append("## Concurrent Burst")
    lines.append("")
    burst = results["concurrent_burst"]
    lines.append(f"- {burst['description']}")
    lines.append(_render_mode_line("Direct", burst["runs"]["direct"]))
    lines.append(_render_mode_line("ByteAI Cache hybrid", burst["runs"]["hybrid"]))
    lines.append("")
    lines.append("## Routing Blend")
    lines.append("")
    routing = results["routing_blend"]
    lines.append(f"- {routing['description']}")
    lines.append(
        _render_mode_line(
            "Direct expensive",
            routing["runs"]["direct_expensive"],
            baseline_label="direct_expensive",
        )
    )
    lines.append(
        _render_mode_line(
            "ByteAI Cache hybrid fixed-expensive",
            routing["runs"]["byte_hybrid_expensive"],
            baseline_label="direct_expensive",
        )
    )
    lines.append(
        _render_mode_line(
            "ByteAI Cache hybrid routed",
            routing["runs"]["byte_hybrid_routed"],
            baseline_label="direct_expensive",
        )
    )
    lines.append("")
    lines.append("## Memory Snapshot")
    lines.append("")
    for line in _render_memory_block(routing["runs"]["byte_hybrid_routed"].get("memory", {})):
        lines.append(line)
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- Coding workloads benefit most from normalized and hybrid reuse; broad semantic reuse is intentionally not the main lever because code answers are high-risk to share loosely."
    )
    lines.append(
        "- Bug-fix and test-generation prompts show whether ByteAI Cache can save money when the same selected code gets wrapped in slightly different editor instructions."
    )
    lines.append(
        "- Prewarming is especially useful for known hot diagnostics or common helper functions inside an editor product."
    )
    lines.append(
        "- The routing blend answers the product question for Cursor-like traffic: ByteAI Cache can still save money on coding requests even when a request misses cache, as long as the easy coding asks are routed to a cheaper model."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--report", default="docs/reports/openai_cursor_coding_benchmark.md")
    parser.add_argument("--json-report", default="docs/reports/openai_cursor_coding_benchmark.json")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("BYTE_TEST_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Set BYTE_TEST_OPENAI_API_KEY or pass --api-key.")

    results: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark_model": CHEAP_MODEL,
        "expensive_model": EXPENSIVE_MODEL,
        "pricing": PRICING,
        "sequential_scenarios": [],
    }

    for scenario in _build_sequential_scenarios():
        runs = {"direct": _run_direct_sequence(api_key, scenario["items"], model=CHEAP_MODEL)}
        warm_data = scenario.get("warm_data")
        for mode in BYTE_MODES:
            runs[mode] = _run_byte_sequence(
                api_key,
                scenario["items"],
                mode=mode,
                model=CHEAP_MODEL,
                routed=False,
                scenario_name=scenario["name"],
                warm_data=warm_data,
            )
        results["sequential_scenarios"].append(
            _scenario_summary(
                scenario["name"],
                scenario["description"],
                scenario["items"],
                runs,
                baseline_key="direct",
                warm_data=warm_data,
            )
        )

    burst = _build_concurrent_scenario()
    burst_runs = {
        "direct": _run_direct_concurrent(api_key, burst["items"], model=CHEAP_MODEL),
        "hybrid": _run_byte_concurrent(
            api_key,
            burst["items"],
            mode="hybrid",
            model=CHEAP_MODEL,
            scenario_name=burst["name"],
        ),
    }
    baseline_cost = burst_runs["direct"]["total_cost_usd"]
    baseline_latency = burst_runs["direct"]["avg_latency_ms"]
    burst_runs["hybrid"]["saved_vs_baseline_usd"] = round(
        baseline_cost - burst_runs["hybrid"]["total_cost_usd"], 8
    )
    burst_runs["hybrid"]["savings_ratio"] = (
        round((baseline_cost - burst_runs["hybrid"]["total_cost_usd"]) / baseline_cost, 4)
        if baseline_cost
        else 0.0
    )
    burst_runs["hybrid"]["latency_delta_ms"] = round(
        burst_runs["hybrid"]["avg_latency_ms"] - baseline_latency, 2
    )
    burst_runs["direct"]["saved_vs_baseline_usd"] = 0.0
    burst_runs["direct"]["savings_ratio"] = 0.0
    burst_runs["direct"]["latency_delta_ms"] = 0.0
    results["concurrent_burst"] = {
        "name": burst["name"],
        "description": burst["description"],
        "request_count": len(burst["items"]),
        "runs": burst_runs,
    }

    routing = _build_routing_scenario()
    routing_runs = {
        "direct_expensive": _run_direct_sequence(api_key, routing["items"], model=EXPENSIVE_MODEL),
        "byte_hybrid_expensive": _run_byte_sequence(
            api_key,
            routing["items"],
            mode="hybrid",
            model=EXPENSIVE_MODEL,
            routed=False,
            scenario_name=routing["name"] + "::fixed",
        ),
        "byte_hybrid_routed": _run_byte_sequence(
            api_key,
            routing["items"],
            mode="hybrid",
            model=EXPENSIVE_MODEL,
            routed=True,
            scenario_name=routing["name"] + "::routed",
            capture_memory=True,
        ),
    }
    results["routing_blend"] = _scenario_summary(
        routing["name"],
        routing["description"],
        routing["items"],
        routing_runs,
        baseline_key="direct_expensive",
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(results), encoding="utf-8")

    json_path = Path(args.json_report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps({"report": str(report_path), "json_report": str(json_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
