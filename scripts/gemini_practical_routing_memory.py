import argparse
import importlib.util
import json
from typing import Any
from datetime import datetime
from pathlib import Path


def _load_module() -> Any:
    module_path = Path(__file__).with_name("deep_multi_provider_routing_memory.py")
    spec = importlib.util.spec_from_file_location("byte_multi_provider_bench", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()
_ORIGINAL_SCENARIOS = MODULE.HELPERS._build_sequential_scenarios
_KEEP = {"unique_18", "canonical_templates_12", "prewarmed_hotset_8", "mixed_workload_24"}


def _trimmed_scenarios() -> Any:
    return [scenario for scenario in _ORIGINAL_SCENARIOS() if scenario["name"] in _KEEP]


MODULE.HELPERS._build_sequential_scenarios = _trimmed_scenarios
MODULE.BYTE_MODES = ["normalized", "hybrid"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report", default="docs/reports/deep_multi_provider_routing_memory_gemini.md"
    )
    parser.add_argument(
        "--json-report", default="docs/reports/deep_multi_provider_routing_memory_gemini.json"
    )
    args = parser.parse_args()

    spec = MODULE.PROVIDERS["gemini"]
    api_key = MODULE._env_value(spec.api_envs)
    if not api_key:
        raise SystemExit(f"Missing one of: {', '.join(spec.api_envs)}")

    try:
        provider_payload = MODULE._build_provider_payload(spec, api_key)
    except Exception as exc:
        provider_payload = {
            "name": spec.name,
            "display_name": spec.display_name,
            "skipped": True,
            "reason": str(exc),
            "pricing_sources": spec.pricing_sources,
            "compatibility_note": spec.compatibility_note,
        }

    results = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "profile": "practical_gemini",
        "providers": [provider_payload],
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(MODULE._render_report(results), encoding="utf-8")

    json_path = Path(args.json_report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps({"report": str(report_path), "json_report": str(json_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
