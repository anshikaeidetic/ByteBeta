import ast
import re

from byte.processor.pre import normalize_text

_LABEL_PATTERNS = (
    re.compile(r"(?is)(?:labels?|classes?)\s*:\s*(?P<labels>[^\n]+)"),
    re.compile(r"(?is)(?:framework|complexity)?\s*label\s+from\s*\{(?P<labels>[^}]+)\}"),
    re.compile(r"(?is)\bfrom\s*\{(?P<labels>[^}]+)\}"),
)

_FRAMEWORK_KEYWORDS = {
    "PYTEST": ("pytest",),
    "UNITTEST": ("unittest", "unit test", "python unittest"),
    "JEST": ("jest",),
    "VITEST": ("vitest",),
    "RSPEC": ("rspec",),
    "JUNIT": ("junit",),
}

_SIGNAL_MAP = {
    "MUTABLE_DEFAULT": ("mutable default", "mutable default argument"),
    "OFF_BY_ONE": ("off by one", "off-by-one", "loop bound"),
    "SYNTAX_ERROR": ("syntax error", "invalid syntax", "syntaxerror"),
    "BROAD_EXCEPTION": ("broad exception", "bare except"),
}

_SYNTHETIC_TOKEN_PATTERN = re.compile(r"^[A-Z0-9_.:-]{3,}$")


def extract_label_candidates(request_text: str) -> list[str]:
    labels: list[str] = []
    for pattern in _LABEL_PATTERNS:
        match = pattern.search(request_text or "")
        if not match:
            continue
        labels.extend(_split_label_values(match.group("labels")))
    ordered = []
    seen = set()
    for item in labels:
        key = item.upper()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def infer_coding_label_from_request(
    *,
    category: str,
    request_text: str,
    labels: list[str],
    style: str = "",
) -> str:
    normalized_category = str(category or "").strip()
    if normalized_category == "code_explanation" and str(style or "").strip() == "complexity":
        return infer_complexity_label_from_request(request_text, labels)
    if normalized_category == "test_generation":
        return infer_framework_label_from_request(request_text, labels)
    if normalized_category == "code_fix":
        return infer_bug_label_from_request(request_text, labels)
    return ""


def infer_complexity_label_from_request(request_text: str, labels: list[str]) -> str:
    code = extract_code_block(request_text)
    if not code:
        return ""
    inferred = ""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None
    if tree is not None:
        loop_depth = _max_iteration_depth(tree)
        if loop_depth >= 2:
            inferred = "O_N_SQUARED"
        elif loop_depth == 1:
            inferred = "O_N"
        else:
            inferred = "O_1"
    else:
        inferred = _infer_complexity_label_from_code_fallback(code)
    return _pick_label(labels, inferred)


def infer_framework_label_from_request(request_text: str, labels: list[str]) -> str:
    normalized = normalize_text(request_text)
    code = normalize_text(extract_code_block(request_text))
    for label, keywords in _FRAMEWORK_KEYWORDS.items():
        candidate = _pick_label(labels, label)
        if not candidate:
            continue
        if any(_contains_normalized_phrase(normalized, keyword) for keyword in keywords):
            return candidate
        if code and any(_contains_normalized_phrase(code, keyword) for keyword in keywords):
            return candidate
    return ""


def infer_bug_label_from_request(request_text: str, labels: list[str]) -> str:
    normalized = normalize_text(request_text)
    for label, signals in _SIGNAL_MAP.items():
        candidate = _pick_label(labels, label)
        if not candidate:
            continue
        if any(_contains_normalized_phrase(normalized, signal) for signal in signals):
            return candidate
    code = extract_code_block(request_text)
    if not code:
        return ""
    if _pick_label(labels, "SYNTAX_ERROR"):
        try:
            ast.parse(code)
        except SyntaxError:
            return _pick_label(labels, "SYNTAX_ERROR")
    if _pick_label(labels, "MUTABLE_DEFAULT") and re.search(
        r"(?m)^\s*def\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*=\s*(?:\[\]|\{\}|set\(\)|dict\(\))",
        code,
    ):
        return _pick_label(labels, "MUTABLE_DEFAULT")
    if _pick_label(labels, "BROAD_EXCEPTION") and re.search(
        r"(?m)^\s*except\s*(?::|Exception\s*:)",
        code,
    ):
        return _pick_label(labels, "BROAD_EXCEPTION")
    if _pick_label(labels, "OFF_BY_ONE") and _looks_like_off_by_one(code):
        return _pick_label(labels, "OFF_BY_ONE")
    return ""


def supports_coding_exact_contract(category: str, token: str) -> bool:
    if str(category or "").strip() not in {
        "code_fix",
        "code_explanation",
        "test_generation",
        "documentation",
        "code_refactor",
    }:
        return False
    candidate = str(token or "").strip()
    if not candidate or not _SYNTHETIC_TOKEN_PATTERN.match(candidate):
        return False
    return any(marker in candidate for marker in ("_", ":", ".", "-"))


def extract_code_block(request_text: str) -> str:
    match = re.search(r"(?is)```(?:[a-z0-9_+#./-]*)\s*\n(?P<code>.+?)```", request_text or "")
    if not match:
        return ""
    return str(match.group("code") or "").strip()


def _looks_like_off_by_one(code: str) -> bool:
    return any(
        re.search(pattern, code, flags=re.I | re.M) is not None
        for pattern in (
            r"range\s*\(\s*len\s*\([^)]+\)\s*\+\s*1\s*\)",
            r"range\s*\(\s*0\s*,\s*len\s*\([^)]+\)\s*\+\s*1\s*\)",
            r"<=\s*len\s*\(",
        )
    )


def _split_label_values(raw: str) -> list[str]:
    parts = re.split(r"[,/|;\n]", raw or "")
    values = []
    for part in parts:
        candidate = part.strip().strip("{}[]()")
        if not candidate:
            continue
        if candidate.lower() in {"and", "or"}:
            continue
        values.append(candidate)
    return values


def _contains_normalized_phrase(text: str, phrase: str) -> bool:
    normalized_text = normalize_text(text)
    normalized_phrase = normalize_text(phrase)
    if not normalized_text or not normalized_phrase:
        return False
    return (
        re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_phrase)}(?![a-z0-9])",
            normalized_text,
        )
        is not None
    )


def _infer_complexity_label_from_code_fallback(code: str) -> str:
    lines = [line.rstrip() for line in str(code or "").splitlines() if line.strip()]
    max_depth = 0
    stack: list[int] = []
    for line in lines:
        indent = len(line) - len(line.lstrip(" "))
        while stack and indent <= stack[-1]:
            stack.pop()
        if re.match(r"^\s*(for|while)\b", line):
            stack.append(indent)
            max_depth = max(max_depth, len(stack))
    if max_depth >= 2:
        return "O_N_SQUARED"
    if max_depth == 1:
        return "O_N"
    return "O_1"


def _max_iteration_depth(node: ast.AST, depth: int = 0) -> int:
    current_depth = depth
    if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
        current_depth = depth + 1
    best = current_depth
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        best = max(best, depth + len(getattr(node, "generators", [])))
    for child in ast.iter_child_nodes(node):
        child_depth = (
            current_depth if isinstance(node, (ast.For, ast.AsyncFor, ast.While)) else depth
        )
        best = max(best, _max_iteration_depth(child, child_depth))
    return best


def _pick_label(labels: list[str], canonical_label: str) -> str:
    wanted = str(canonical_label or "").strip().upper()
    for label in labels:
        if str(label or "").strip().upper() == wanted:
            return label
    return ""
