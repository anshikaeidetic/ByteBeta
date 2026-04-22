#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

explicit_python=""
args=()
while (($#)); do
  if [[ "$1" == "--python" && $# -ge 2 ]]; then
    explicit_python="$2"
    args+=("$1" "$2")
    shift 2
    continue
  fi
  args+=("$1")
  shift
done

probe_python() {
  local require_venv="$1"
  shift
  if [[ "$require_venv" == "true" ]]; then
    "$@" -c "import encodings, pip, venv"
  else
    "$@" -c "import encodings, pip"
  fi
}

if [[ -n "$explicit_python" ]]; then
  if ! probe_python false "$explicit_python" >/dev/null 2>&1; then
    echo "Explicit Python interpreter is not usable: $explicit_python" >&2
    exit 1
  fi
  exec "$explicit_python" "$ROOT_DIR/scripts/bootstrap_dev.py" "${args[@]}"
fi

for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 && probe_python true "$candidate" >/dev/null 2>&1; then
    exec "$candidate" "$ROOT_DIR/scripts/bootstrap_dev.py" "${args[@]}"
  fi
done

echo "Unable to locate a healthy Python interpreter for bootstrap." >&2
exit 1
