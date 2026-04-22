"""Refresh repo-owned lockfiles from pyproject extras."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_PLATFORM_VARIABLE_NAMES = {
    "colorama",
    "jeepney",
    "pywin32-ctypes",
    "secretstorage",
    "triton",
}
_PLATFORM_VARIABLE_PREFIXES = ("cuda-", "nvidia-")
LOCK_PIP_ARGS = (
    "--platform manylinux2014_x86_64 "
    "--python-version 3.10 "
    "--implementation cp "
    "--abi cp310 "
    "--only-binary=:all:"
)
_OUTPUT_FILE_PATTERN = re.compile(r"--output-file(?:=|\s+)(?:'[^']+'|\"[^\"]+\"|[^\s]+)")
LOCKS = {
    ROOT
    / "requirements"
    / "dev-ci.txt": (
        "dev",
        "test",
        "openai",
        "groq",
        "sql",
        "faiss",
        "huggingface",
        "langchain",
        "onnx",
        "observability",
        "server",
    ),
    ROOT
    / "requirements"
    / "security-ci.txt": (
        "dev",
        "test",
        "openai",
        "groq",
        "sql",
        "faiss",
        "huggingface",
        "langchain",
        "onnx",
        "security",
        "observability",
        "server",
    ),
}


def relative_lock_path(output_path: Path) -> str:
    return output_path.relative_to(ROOT).as_posix()


def normalize_output_file_reference(contents: str, output_path: Path) -> str:
    normalized_output = f"--output-file='{relative_lock_path(output_path)}'"
    return _OUTPUT_FILE_PATTERN.sub(lambda _: normalized_output, contents)


def _is_platform_variable_requirement(line: str) -> bool:
    if line.startswith((" ", "#")) or "==" not in line:
        return False
    name = line.split("==", 1)[0].strip().lower()
    return name in _PLATFORM_VARIABLE_NAMES or name.startswith(_PLATFORM_VARIABLE_PREFIXES)


def _is_platform_variable_annotation(line: str) -> bool:
    stripped = line.strip().lower()
    if not stripped.startswith("#"):
        return False
    return any(
        f" {name}" in stripped or stripped.endswith(name)
        for name in _PLATFORM_VARIABLE_NAMES
    ) or any(prefix in stripped for prefix in _PLATFORM_VARIABLE_PREFIXES)


def canonicalize_lockfile_contents(contents: str) -> str:
    lines = contents.splitlines(keepends=True)
    canonical: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if _is_platform_variable_annotation(line):
            index += 1
            continue
        if _is_platform_variable_requirement(line):
            index += 1
            while index < len(lines) and (
                lines[index].startswith("    #") or not lines[index].strip()
            ):
                index += 1
            continue
        canonical.append(line)
        index += 1
    return "".join(canonical)


def _compile_lock(
    output_path: Path,
    extras: tuple[str, ...],
    *,
    header_output_path: Path | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        command_output_path = relative_lock_path(output_path)
    except ValueError:
        command_output_path = str(output_path)
    command = [
        sys.executable,
        "-m",
        "piptools",
        "compile",
        "pyproject.toml",
        "--resolver=backtracking",
        "--upgrade",
        "--strip-extras",
        "--newline",
        "lf",
        "--quiet",
        "--output-file",
        command_output_path,
        "--pip-args",
        LOCK_PIP_ARGS,
    ]
    for extra in extras:
        command.extend(["--extra", extra])
    subprocess.run(command, cwd=ROOT, check=True)
    contents = output_path.read_text(encoding="utf-8")
    normalized = normalize_output_file_reference(contents, header_output_path or output_path)
    output_path.write_text(canonicalize_lockfile_contents(normalized), encoding="utf-8")


def main() -> int:
    for output_path, extras in LOCKS.items():
        _compile_lock(output_path, extras)
        print(f"Refreshed {output_path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
