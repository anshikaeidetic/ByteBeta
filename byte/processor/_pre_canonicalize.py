"""Prompt canonicalization helpers extracted from ``byte.processor.pre``."""

import hashlib
import re
import unicodedata
from typing import Any

_EXACT_ANSWER_PATTERNS = [
    re.compile(
        r"\b(?:reply|respond|return|answer|output)\b(?:\s+\w+){0,2}\s+exactly\s+"
        r"(?P<token>[a-z0-9_.-]+(?:\s+[a-z0-9_.-]+){0,2}?)(?=\s+and\s+nothing\s+else\b|$)"
    ),
    re.compile(
        r"\b(?:reply|respond|return|answer|output)\s+only\s+"
        r"(?P<token>[a-z0-9_.-]+(?:\s+[a-z0-9_.-]+){0,2})\b"
    ),
    re.compile(
        r"\bonly\s+(?:reply|respond|return|answer|output)\s+"
        r"(?P<token>[a-z0-9_.-]+(?:\s+[a-z0-9_.-]+){0,2})\b"
    ),
    re.compile(
        r"\bkeep\s+the\s+answer\s+to\s+"
        r"(?P<token>[a-z0-9_.-]+(?:\s+[a-z0-9_.-]+){0,2})\b"
    ),
]

_TRANSLATION_PATTERNS = [
    re.compile(
        r"(?is)\b(?:translate|render)\b.*?\b(?:to|into)\s+"
        r"(?P<language>[a-z][a-z -]{1,32})\s*:\s*(?P<payload>.+)"
    ),
    re.compile(
        r"(?is)(?:text|phrase|sentence)\s*:\s*(?P<payload>.+?)\s+"
        r"(?:translate|render)\s+(?:it\s+)?(?:to|into)\s+"
        r"(?P<language>[a-z][a-z -]{1,32})\b"
    ),
]

_SUMMARIZATION_PATTERNS = [
    re.compile(
        r"(?is)\b(?:summari[sz]e|tl\s*dr|give\s+a\s+summary|write\s+a\s+summary)\b"
        r".*?\b(?:text|article|passage|content|document)\s*:\s*(?P<payload>.+)"
    ),
    re.compile(
        r"(?is)(?:text|article|passage|content|document)\s*:\s*(?P<payload>.+?)\s+"
        r"(?:summari[sz]e|give\s+a\s+summary|write\s+a\s+summary|tl\s*dr)\b"
    ),
]

_EXTRACTION_PATTERNS = [
    re.compile(
        r"(?is)\b(?:extract|return)\b.*?\b(?:fields?|keys?)\s*:\s*(?P<fields>[^\n]+)"
        r".*?\b(?:text|record|input|document|content|review|message|ticket|clause|incident|email)\s*:\s*(?P<payload>.+)"
    ),
    re.compile(
        r"(?is)(?:text|record|input|document|content|review|message|ticket|clause|incident|email)\s*:\s*(?P<payload>.+?)\s+"
        r"(?:extract|return)\b.*?\b(?:fields?|keys?)\s*:\s*(?P<fields>[^\n]+)"
    ),
    re.compile(
        r"(?is)\b(?:extract|return)\b.*?\b(?:fields?|keys?)\b\s*(?P<fields>\"[^\"]+\"(?:\s*(?:,|and)\s*\"[^\"]+\")*)"
        r".*?\b(?:text|record|input|document|content|review|message|ticket|clause|incident|email)\s*:\s*(?P<payload>.+)"
    ),
    re.compile(
        r"(?is)(?:text|record|input|document|content|review|message|ticket|clause|incident|email)\s*:\s*(?P<payload>.+?)\s+"
        r"(?:extract|return)\b.*?\b(?:fields?|keys?)\b\s*(?P<fields>\"[^\"]+\"(?:\s*(?:,|and)\s*\"[^\"]+\")*)"
    ),
]

_CODE_BLOCK_PATTERN = re.compile(r"(?is)```(?P<language>[a-z0-9_+#./-]*)\s*\n(?P<code>.+?)```")

_RELEVANCE_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "using",
    "use",
    "question",
    "answer",
    "context",
    "retrieval",
    "available",
    "docs",
    "doc",
    "document",
    "documents",
    "support",
    "article",
    "articles",
    "repo",
    "workspace",
}

_INLINE_PATH_PATTERN = re.compile(
    r"(?im)\b[a-z]:[\\/][^\s:]+(?:\.\w+)?(?::\d+)?\b|\b(?:src|app|lib|tests?|packages?)[\\/][^\s:]+(?:\.\w+)?(?::\d+)?\b"
)

_CODE_LINE_PATTERN = re.compile(r"(?im)\b(?:line|column|lines?)\s+\d+(?:\s*-\s*\d+)?\b")

_CODE_CONTEXT_HEADER_PATTERN = re.compile(
    r"(?im)^\s*(?:file|path|selection|selected code|cursor|symbol)\s*:\s*[^\n]*$"
)

_CODE_DIAGNOSTIC_LABELS = (
    ("mutable default", "mutable_default"),
    ("off by one", "off_by_one"),
    ("syntax error", "syntax_error"),
    ("invalid syntax", "syntax_error"),
    ("syntaxerror", "syntax_error"),
    ("type error", "type_error"),
    ("typeerror", "type_error"),
    ("attribute error", "attribute_error"),
    ("attributeerror", "attribute_error"),
    ("name error", "name_error"),
    ("nameerror", "name_error"),
    ("missing await", "missing_await"),
    ("lint", "lint"),
    ("failing test", "failing_test"),
    ("test failure", "failing_test"),
    ("traceback", "traceback"),
    ("exception", "exception"),
    ("broad exception", "broad_exception"),
    ("null pointer", "null_pointer"),
    ("npe", "null_pointer"),
)

_CODE_TEST_FRAMEWORKS = (
    ("pytest", "pytest"),
    ("unittest", "unittest"),
    ("jest", "jest"),
    ("vitest", "vitest"),
    ("rspec", "rspec"),
    ("junit", "junit"),
)

_GENERIC_EXACT_TOKENS = {
    "one label",
    "single label",
    "one word",
    "single word",
    "one number",
    "single number",
    "just one label",
    "just one word",
    "digits only",
    "just digits",
}


def normalize_text(text: Any) -> str:
    """Normalize text for opt-in cache-key reuse."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _contains_normalized_phrase(normalized: str, phrase: str) -> bool:
    normalized_text = f" {str(normalized or '').strip().lower()} "
    normalized_phrase = str(phrase or "").strip().lower()
    if not normalized_phrase:
        return False
    return f" {normalized_phrase} " in normalized_text


def _normalize_payload_text(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def _payload_digest(text: str) -> str:
    normalized_payload = _normalize_payload_text(text)
    if not normalized_payload:
        return ""
    return hashlib.sha256(normalized_payload.encode("utf-8")).hexdigest()[:16]


def _normalize_code_text(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _code_digest(text: str) -> str:
    normalized_code = _normalize_code_text(text)
    if not normalized_code:
        return ""
    return hashlib.sha256(normalized_code.encode("utf-8")).hexdigest()[:16]


def _extract_code_context(raw_text: str) -> tuple:
    match = _CODE_BLOCK_PATTERN.search(raw_text or "")
    if not match:
        return "", ""
    language = normalize_text(match.group("language")) or "plain"
    code = _normalize_code_text(match.group("code"))
    return language, code


def _normalize_code_shell(raw_text: str) -> str:
    shell = _CODE_BLOCK_PATTERN.sub(" <code> ", raw_text or "")
    shell = _CODE_CONTEXT_HEADER_PATTERN.sub(" ", shell)
    shell = _INLINE_PATH_PATTERN.sub("<path>", shell)
    shell = _CODE_LINE_PATTERN.sub("<line>", shell)
    return normalize_text(shell)


def _code_goal(normalized: str) -> str:
    if any(token in normalized for token in ("readability", "clean up", "cleanup", "clearer")):
        return "readability"
    if any(token in normalized for token in ("performan", "optimiz", "faster")):
        return "performance"
    if any(token in normalized for token in ("type hint", "typing", "annotation")):
        return "typing"
    if any(token in normalized for token in ("rename", "naming")):
        return "naming"
    if "simplify" in normalized:
        return "simplify"
    return "general"


def _code_framework(normalized: str) -> str:
    for needle, label in _CODE_TEST_FRAMEWORKS:
        if needle in normalized:
            return label
    return "generic"


def _code_diagnostic_label(normalized: str) -> str:
    for needle, label in _CODE_DIAGNOSTIC_LABELS:
        if needle in normalized:
            return label
    return "general"


def _canonicalize_exact_answer(normalized: str) -> str:
    for pattern in _EXACT_ANSWER_PATTERNS:
        match = pattern.search(normalized)
        if match:
            token = match.group("token").strip()
            if token and token not in _GENERIC_EXACT_TOKENS:
                return f"exact_answer::{token}"
    return ""


def _canonicalize_capital_question(normalized: str) -> str:
    if "capital" not in normalized:
        return ""
    patterns = (
        re.compile(
            r"(?is)\bwhat\s+is\s+the\s+capital(?:\s+city)?\s+of\s+(?P<entity>[a-z][a-z .'-]{1,60})"
        ),
        re.compile(
            r"(?is)\bwhich\s+city\s+is\s+the\s+capital(?:\s+city)?\s+of\s+(?P<entity>[a-z][a-z .'-]{1,60})"
        ),
        re.compile(
            r"(?is)\bname\s+the\s+capital(?:\s+city)?\s+of\s+(?P<entity>[a-z][a-z .'-]{1,60})"
        ),
    )
    for pattern in patterns:
        match = pattern.search(normalized)
        if not match:
            continue
        entity = normalize_text(match.group("entity"))
        entity = re.split(
            r"\b(?:please|reply|answer|return|only|today|currently|now)\b",
            entity,
            maxsplit=1,
        )[0].strip()
        words = [word for word in entity.split() if word not in {"the", "a", "an"}]
        if not words or len(words) > 5:
            continue
        return f"qa_fact::capital::{'_'.join(words)}"
    return ""


def _normalize_label_set(labels: str) -> str:
    cleaned = unicodedata.normalize("NFKC", labels).lower()
    if not cleaned:
        return ""
    cleaned = re.sub(r"\bor\b|\band\b", ",", cleaned)
    cleaned = re.sub(r"[;/|]", ",", cleaned)
    cleaned = re.sub(
        r"\b(?:or|and|only|exactly|reply|respond|answer|return|with|one|label)\b",
        " ",
        cleaned,
    )
    parts = cleaned.split(",")
    normalized_parts = sorted({normalize_text(part) for part in parts if normalize_text(part)})
    return "|".join(normalized_parts)


def _normalize_field_set(fields: str) -> str:
    cleaned = unicodedata.normalize("NFKC", fields).lower()
    if not cleaned:
        return ""
    cleaned = re.sub(r"\bor\b|\band\b", ",", cleaned)
    cleaned = re.sub(r"[;/|]", ",", cleaned)
    cleaned = re.sub(
        r"\b(?:return|extract|field|fields|key|keys|json|yaml|csv|only|with)\b",
        " ",
        cleaned,
    )
    parts = cleaned.split(",")
    normalized_parts = sorted({normalize_text(part) for part in parts if normalize_text(part)})
    return "|".join(normalized_parts)


def _summary_style(normalized: str) -> str:
    if "one sentence" in normalized or "single sentence" in normalized:
        return "one_sentence"
    if "bullet" in normalized:
        return "bullets"
    if "headline" in normalized:
        return "headline"
    if "brief" in normalized or "concise" in normalized:
        return "concise"
    return "default"


def _extract_format(normalized: str) -> str:
    if "json" in normalized:
        return "json"
    if "yaml" in normalized:
        return "yaml"
    if "csv" in normalized:
        return "csv"
    return "plain"


def _canonicalize_labeled_classification(raw_text: str, normalized: str) -> str:
    if not re.search(
        r"\b(?:classif(?:y|ication)?|sentiment|triag(?:e|ing)?|labels?|classes?|categories?)\b",
        normalized,
    ):
        return ""
    labels_match = re.search(
        r"(?is)(?:labels?|classes?|categories?)\s*:\s*(?P<labels>[^\n]+)", raw_text
    )
    if not labels_match:
        labels_match = re.search(
            r"(?is)\b(?:one|single)\s+(?:label|class|category)\s+(?:from|out\s+of)\s*\{(?P<labels>[^}]+)\}",
            raw_text,
        )
    if not labels_match:
        labels_match = re.search(
            r"(?is)\b(?:choose|pick|reply|answer|return)\b.*?\bfrom\s*\{(?P<labels>[^}]+)\}",
            raw_text,
        )
    payload_match = re.search(
        r"(?is)(?P<payload_key>text|review|message|input|ticket|document|record|clause|incident|email)\s*:\s*"
        r"(?P<payload>\"[^\"]+\"|[^\n]+)",
        raw_text,
    )
    if not labels_match or not payload_match:
        return ""

    labels = _normalize_label_set(labels_match.group("labels"))
    payload_key = normalize_text(payload_match.group("payload_key"))
    payload = payload_match.group("payload").strip().strip('"')
    payload_digest = _payload_digest(payload)
    if labels and payload_digest:
        return f"classify::{payload_key}::{labels}::{payload_digest}"
    return ""


def _canonicalize_translation(raw_text: str, normalized: str) -> str:
    if "translate" not in normalized and "render" not in normalized:
        return ""
    for pattern in _TRANSLATION_PATTERNS:
        match = pattern.search(raw_text)
        if not match:
            continue
        language = normalize_text(match.group("language"))
        payload_digest = _payload_digest(match.group("payload"))
        if language and payload_digest:
            return f"translate::{language}::{payload_digest}"
    return ""


def _canonicalize_summarization(raw_text: str, normalized: str) -> str:
    if not any(token in normalized for token in ("summarize", "summary", "tldr", "tl dr")):
        return ""
    style = _summary_style(normalized)
    for pattern in _SUMMARIZATION_PATTERNS:
        match = pattern.search(raw_text)
        if not match:
            continue
        payload_digest = _payload_digest(match.group("payload"))
        if payload_digest:
            return f"summarize::{style}::{payload_digest}"
    return ""


def _canonicalize_extraction(raw_text: str, normalized: str) -> str:
    if not any(token in normalized for token in ("extract", "fields", "keys", "return json")):
        return ""
    fmt = _extract_format(normalized)
    for pattern in _EXTRACTION_PATTERNS:
        match = pattern.search(raw_text)
        if not match:
            continue
        fields = _normalize_field_set(match.group("fields"))
        payload = (
            re.split(
                r"(?is)\n+\s*(?:return|extract|output|answer)\b",
                match.group("payload"),
                maxsplit=1,
            )[0]
            .strip()
            .strip('"')
        )
        payload_digest = _payload_digest(payload)
        if fields and payload_digest:
            return f"extract::{fmt}::{fields}::{payload_digest}"
    return ""


def _canonicalize_code_fix(raw_text: str, normalized: str) -> str:
    if not any(
        token in normalized
        for token in (
            "fix",
            "bug",
            "error",
            "traceback",
            "diagnostic",
            "lint",
            "failing test",
            "exception",
        )
    ):
        return ""
    language, code = _extract_code_context(raw_text)
    code_fingerprint = _code_digest(code)
    if not code_fingerprint:
        return ""
    diagnostic = _code_diagnostic_label(_normalize_code_shell(raw_text))
    return f"code_fix::{language}::{diagnostic}::{code_fingerprint}"


def _canonicalize_code_tests(raw_text: str, normalized: str) -> str:
    if not any(
        token in normalized
        for token in (
            "unit test",
            "unit tests",
            "write tests",
            "add tests",
            "test case",
            "pytest",
            "unittest",
            "jest",
            "vitest",
        )
    ):
        return ""
    language, code = _extract_code_context(raw_text)
    code_fingerprint = _code_digest(code)
    if not code_fingerprint:
        return ""
    framework = _code_framework(normalized)
    return f"code_tests::{language}::{framework}::{code_fingerprint}"


def _canonicalize_code_explanation(raw_text: str, normalized: str) -> str:
    if not any(
        token in normalized
        for token in (
            "explain this code",
            "explain this function",
            "explain the code",
            "explain the function",
            "what does this code do",
            "what does this function do",
            "walk me through",
            "time complexity",
            "big o",
        )
    ):
        return ""
    language, code = _extract_code_context(raw_text)
    code_fingerprint = _code_digest(code)
    if not code_fingerprint:
        return ""
    if any(token in normalized for token in ("complexity label", "time complexity", "big o")):
        style = "complexity"
    else:
        style = _summary_style(normalized)
    return f"code_explain::{language}::{style}::{code_fingerprint}"


def _canonicalize_code_refactor(raw_text: str, normalized: str) -> str:
    refactor_hints = (
        "refactor",
        "clean up",
        "cleanup code",
        "cleanup this",
        "cleanup the",
        "improve readability",
        "optimize this code",
        "simplify",
    )
    if not any(_contains_normalized_phrase(normalized, token) for token in refactor_hints):
        return ""
    language, code = _extract_code_context(raw_text)
    code_fingerprint = _code_digest(code)
    if not code_fingerprint:
        return ""
    goal = _code_goal(normalized)
    return f"code_refactor::{language}::{goal}::{code_fingerprint}"


def _canonicalize_code_docstring(raw_text: str, normalized: str) -> str:
    if not any(
        token in normalized for token in ("docstring", "documentation comment", "jsdoc", "add docs")
    ):
        return ""
    language, code = _extract_code_context(raw_text)
    code_fingerprint = _code_digest(code)
    if not code_fingerprint:
        return ""
    return f"code_doc::{language}::{code_fingerprint}"


def canonicalize_text(text: Any) -> str:
    """Canonicalize common prompt templates into stable request-class keys."""
    raw_text = "" if text is None else str(text)
    normalized = normalize_text(raw_text)
    if not normalized:
        return normalized

    canonical = _canonicalize_labeled_classification(raw_text, normalized)
    if canonical:
        return canonical

    canonical = _canonicalize_translation(raw_text, normalized)
    if canonical:
        return canonical

    canonical = _canonicalize_code_fix(raw_text, normalized)
    if canonical:
        return canonical

    canonical = _canonicalize_code_tests(raw_text, normalized)
    if canonical:
        return canonical

    canonical = _canonicalize_code_explanation(raw_text, normalized)
    if canonical:
        return canonical

    canonical = _canonicalize_code_docstring(raw_text, normalized)
    if canonical:
        return canonical

    canonical = _canonicalize_code_refactor(raw_text, normalized)
    if canonical:
        return canonical

    canonical = _canonicalize_summarization(raw_text, normalized)
    if canonical:
        return canonical

    canonical = _canonicalize_extraction(raw_text, normalized)
    if canonical:
        return canonical

    canonical = _canonicalize_exact_answer(normalized)
    if canonical:
        return canonical

    canonical = _canonicalize_capital_question(normalized)
    if canonical:
        return canonical

    return normalized


__all__ = [name for name in globals() if not name.startswith("__")]
