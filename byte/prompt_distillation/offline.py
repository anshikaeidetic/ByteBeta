from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .core import export_prompt_distillation_manifest


def write_prompt_distillation_manifest(
    requests: list[dict[str, Any]],
    *,
    out_path: str,
    artifact_version: str = "byte-prompt-distill-v1",
    signing_key: str = "",
) -> dict[str, Any]:
    manifest = export_prompt_distillation_manifest(
        requests,
        artifact_version=artifact_version,
        signing_key=signing_key,
    )
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return {"path": str(target), "module_count": int(manifest.get("module_count", 0) or 0)}
