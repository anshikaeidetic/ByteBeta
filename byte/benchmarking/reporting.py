"""Benchmark report writers and renderers for local and release proof bundles."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def write_report_bundle(results: dict[str, Any], out_dir: str) -> dict[str, str]:
    """Write the machine and human benchmark report bundle for one run."""

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    engineering_json = target / "engineering_report.json"
    engineering_details_json = target / "engineering_details.json"
    engineering_md = target / "engineering_report.md"
    executive_md = target / "executive_summary.md"
    reasoning_trace_json = target / "reasoning_graph.json"
    reasoning_graph_svg = target / "reasoning_graph.svg"
    reasoning_graph_html = target / "reasoning_graph.html"
    _write_json(engineering_json, build_engineering_report(results))
    _write_json(engineering_details_json, results)
    _write_text(engineering_md, render_engineering_markdown(results))
    _write_text(executive_md, render_executive_summary(results))
    trace = build_reasoning_trace(results)
    _write_json(reasoning_trace_json, trace)
    _write_text(reasoning_graph_svg, render_reasoning_svg(trace))
    _write_text(reasoning_graph_html, render_reasoning_html(trace))
    return {
        "engineering_json": str(engineering_json),
        "engineering_details_json": str(engineering_details_json),
        "engineering_markdown": str(engineering_md),
        "executive_markdown": str(executive_md),
        "reasoning_trace_json": str(reasoning_trace_json),
        "reasoning_graph_svg": str(reasoning_graph_svg),
        "reasoning_graph_html": str(reasoning_graph_html),
    }


def build_engineering_report(results: dict[str, Any]) -> dict[str, Any]:
    """Build a compact canonical JSON report that preserves phase summaries and raw records."""

    if _has_nested_provider_tree(results):
        phase_summaries, records = _flatten_phase_data(results)
    else:
        phase_summaries = [dict(summary) for summary in (results.get("phase_summaries", []) or [])]
        records = [dict(record) for record in (results.get("records", []) or [])]
    return {
        "schema_version": str(results.get("schema_version", "") or ""),
        "run_id": str(results.get("run_id", "") or ""),
        "profile": str(results.get("profile", "") or ""),
        "profile_base": str(results.get("profile_base", "") or ""),
        "benchmark_track": str(results.get("benchmark_track", "") or ""),
        "execution_mode": str(results.get("execution_mode", "") or ""),
        "generated_at": str(results.get("generated_at", "") or ""),
        "corpus_version": str(results.get("corpus_version", "") or ""),
        "benchmark_contract_version": str(results.get("benchmark_contract_version", "") or ""),
        "scoring_version": str(results.get("scoring_version", "") or ""),
        "trust_policy_version": str(results.get("trust_policy_version", "") or ""),
        "scorecard_mode": str(results.get("scorecard_mode", "") or ""),
        "replicates": int(results.get("replicates", 0) or 0),
        "confidence_level": float(results.get("confidence_level", 0.95) or 0.95),
        "judge_mode": str(results.get("judge_mode", "") or ""),
        "contamination_check": bool(results.get("contamination_check", False)),
        "live_cutoff_date": str(results.get("live_cutoff_date", "") or ""),
        "release_gate": bool(results.get("release_gate", False)),
        "providers_requested": list(results.get("providers_requested", []) or []),
        "providers_executed": list(results.get("providers_executed", []) or []),
        "systems_requested": list(results.get("systems_requested", []) or []),
        "systems_executed": list(results.get("systems_executed", []) or []),
        "manifest_versions": list(results.get("manifest_versions", []) or []),
        "gate_results": list(results.get("gate_results", []) or []),
        "phase_summaries": phase_summaries,
        "records": records,
    }


def render_engineering_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Byte Benchmark Engineering Report",
        "",
        f"- Profile: {results.get('profile', '')}",
        f"- Benchmark track: {results.get('benchmark_track', '')}",
        f"- Execution mode: {results.get('execution_mode', '')}",
        f"- Generated at: {results.get('generated_at', '')}",
        f"- Run id: {results.get('run_id', '')}",
        f"- Corpus version: {results.get('corpus_version', '')}",
        f"- Benchmark contract version: {results.get('benchmark_contract_version', '')}",
        f"- Scoring version: {results.get('scoring_version', '')}",
        f"- Trust policy version: {results.get('trust_policy_version', '')}",
        f"- Scorecard mode: {results.get('scorecard_mode', '')}",
        f"- Replicates: {results.get('replicates', 0)}",
        f"- Confidence level: {float(results.get('confidence_level', 0.95) or 0.95):.2f}",
        f"- Judge mode: {results.get('judge_mode', '')}",
        f"- Contamination check: {bool(results.get('contamination_check', False))}",
        f"- Live cutoff date: {results.get('live_cutoff_date', '') or 'n/a'}",
        f"- Providers executed: {', '.join(results.get('providers_executed', []))}",
        f"- Systems executed: {', '.join(results.get('systems_executed', []))}",
        "",
        "## Gate Summary",
        "",
        "| Gate | Status | Value | Threshold |",
        "| --- | --- | ---: | ---: |",
    ]
    for gate in results.get("gate_results", []):
        lines.append(
            f"| {gate['name']} | {'PASS' if gate['passed'] else 'FAIL'} | {gate['value']} | {gate['threshold']} |"
        )
    lines.extend(["", "## Provider Results", ""])
    for provider, provider_payload in (results.get("providers", {}) or {}).items():
        lines.append(f"### {provider}")
        lines.append("")
        lines.append(
            "| System | Phase | Forced Accuracy | Forced CI 95 | Selective Accuracy | Coverage | Selective CI 95 | Avg Latency | Cost | False Reuse | Confidence ECE | Contamination |"
        )
        lines.append("| --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |")
        for system_name, system_payload in (provider_payload.get("systems", {}) or {}).items():
            for phase_name, phase_payload in (system_payload.get("phases", {}) or {}).items():
                summary = dict(phase_payload.get("summary", {}) or {})
                ci_payload = dict(summary.get("ci_95", {}) or {})
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            system_name,
                            phase_name,
                            f"{float(summary.get('forced_answer_accuracy', 0.0) or 0.0):.4f}",
                            _format_ci(ci_payload.get("forced_answer_accuracy")),
                            f"{float(summary.get('selective_accuracy', 0.0) or 0.0):.4f}",
                            f"{float(summary.get('coverage', 0.0) or 0.0):.4f}",
                            _format_ci(ci_payload.get("selective_accuracy")),
                            f"{float(summary.get('avg_latency_ms', 0.0) or 0.0):.2f}",
                            f"{float(summary.get('cost_usd', 0.0) or 0.0):.6f}",
                            f"{float(summary.get('false_reuse_rate', 0.0) or 0.0):.4f}",
                            f"{float(summary.get('confidence_ece', 0.0) or 0.0):.4f}",
                            str(summary.get("contamination_status", "") or "unknown"),
                        ]
                    )
                    + " |"
                )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_executive_summary(results: dict[str, Any]) -> str:
    highlights = list(results.get("highlights", []) or [])
    cost_table = results.get("dollar_impact", {}) or {}
    lines = [
        "# Byte Benchmark Executive Summary",
        "",
        f"Profile: {results.get('profile', '')}",
        f"Benchmark track: {results.get('benchmark_track', '')}",
        f"Execution mode: {results.get('execution_mode', '')}",
        f"Corpus version: {results.get('corpus_version', '')}",
        f"Benchmark contract version: {results.get('benchmark_contract_version', '')}",
        f"Scoring version: {results.get('scoring_version', '')}",
        f"Trust policy version: {results.get('trust_policy_version', '')}",
        f"Scorecard mode: {results.get('scorecard_mode', '')}",
        "",
        "## Headline Findings",
        "",
    ]
    for highlight in highlights[:8]:
        lines.append(f"- {highlight}")
    lines.extend(
        [
            "",
            "## DeepSeek Scorecards",
            "",
            "| Provider | Phase | Scorecard | Accuracy | Coverage | Sample Size | CI 95 | Contamination |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for provider, provider_payload in (results.get("providers", {}) or {}).items():
        byte_phase = _preferred_phase(provider_payload, "byte")
        if not byte_phase:
            continue
        summary = dict(byte_phase.get("summary", {}) or {})
        ci_payload = dict(summary.get("ci_95", {}) or {})
        lines.append(
            f"| {provider} | {_preferred_phase_name(provider_payload, 'byte')} | Forced | "
            f"{float(summary.get('forced_answer_accuracy', 0.0) or 0.0):.4f} | 1.0000 | "
            f"{int(summary.get('sample_size', 0) or 0)} | {_format_ci(ci_payload.get('forced_answer_accuracy'))} | "
            f"{summary.get('contamination_status', '') or 'unknown'!s} |"
        )
        lines.append(
            f"| {provider} | {_preferred_phase_name(provider_payload, 'byte')} | Selective | "
            f"{float(summary.get('selective_accuracy', 0.0) or 0.0):.4f} | "
            f"{float(summary.get('coverage', 0.0) or 0.0):.4f} | {int(summary.get('answered_count', 0) or 0)} | "
            f"{_format_ci(ci_payload.get('selective_accuracy'))} | "
            f"{summary.get('contamination_status', '') or 'unknown'!s} |"
        )
    lines.extend(
        [
            "",
            "## Prompt Distillation",
            "",
            "| Provider | Prompt Reduction | Faithfulness | Module Reuse |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for provider, provider_payload in (results.get("providers", {}) or {}).items():
        systems = provider_payload.get("systems", {}) or {}
        byte_warm = (
            ((systems.get("byte", {}) or {}).get("phases", {}) or {}).get("warm_100")
            or ((systems.get("byte", {}) or {}).get("phases", {}) or {}).get("warm_1000")
            or ((systems.get("byte", {}) or {}).get("phases", {}) or {}).get("cold")
            or {}
        )
        summary = dict(byte_warm.get("summary", {}) or {})
        if not summary:
            continue
        lines.append(
            f"| {provider} | {float(summary.get('prompt_token_reduction_ratio', 0.0) or 0.0):.4f} | "
            f"{float(summary.get('faithfulness_pass_rate', 0.0) or 0.0):.4f} | "
            f"{float(summary.get('module_reuse_rate', 0.0) or 0.0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Dollar Impact",
            "",
            "| Volume | Direct | Byte | Savings |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for volume, payload in cost_table.items():
        lines.append(
            f"| {volume} | {payload['direct_cost_usd']:.2f} | {payload['byte_cost_usd']:.2f} | {payload['savings_usd']:.2f} |"
        )
    return "\n".join(lines).strip() + "\n"


def build_reasoning_trace(results: dict[str, Any]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    links: list[dict[str, Any]] = []
    for provider, provider_payload in (results.get("providers", {}) or {}).items():
        systems = provider_payload.get("systems", {}) or {}
        for system_name, system_payload in systems.items():
            for phase_name, phase_payload in (system_payload.get("phases", {}) or {}).items():
                for family_name, family_summary in (phase_payload.get("families", {}) or {}).items():
                    family_node = f"{provider}:{system_name}:{phase_name}:{family_name}"
                    _ensure_node(nodes, family_node, family_name)
                    served_counts = dict((family_summary.get("summary", {}) or {}).get("served_via_counts", {}) or {})
                    for mode_name, value in served_counts.items():
                        mode_node = f"{provider}:{system_name}:{phase_name}:{mode_name}"
                        _ensure_node(nodes, mode_node, mode_name)
                        links.append({"source": family_node, "target": mode_node, "value": int(value or 0)})
    return {"nodes": list(nodes.values()), "links": links}


def render_reasoning_svg(trace: dict[str, Any]) -> str:
    nodes = list(trace.get("nodes", []) or [])
    links = list(trace.get("links", []) or [])
    width = 1200
    height = max(360, 40 * max(len(nodes), 6))
    left_x = 70
    right_x = 760
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Georgia,serif;font-size:12px;fill:#102030}.node{fill:#d7e7f5;stroke:#365977;stroke-width:1.2}.flow{stroke:#7aa6c2;stroke-opacity:0.65;fill:none}</style>',
    ]
    family_nodes = [
        node
        for node in nodes
        if node["id"].split(":")[-1] not in {"reuse", "local_compute", "upstream"}
    ]
    mode_nodes = [node for node in nodes if node["id"].split(":")[-1] in {"reuse", "local_compute", "upstream"}]
    family_y = {node["id"]: 40 + index * 36 for index, node in enumerate(family_nodes)}
    mode_y = {node["id"]: 70 + index * 80 for index, node in enumerate(mode_nodes)}
    for node in family_nodes:
        y = family_y[node["id"]]
        svg_lines.append(f'<rect class="node" x="{left_x}" y="{y}" width="240" height="22" rx="5"/>')
        svg_lines.append(f'<text x="{left_x + 10}" y="{y + 15}">{_escape_xml(node["label"])}</text>')
    for node in mode_nodes:
        y = mode_y[node["id"]]
        svg_lines.append(f'<rect class="node" x="{right_x}" y="{y}" width="200" height="26" rx="5"/>')
        svg_lines.append(f'<text x="{right_x + 10}" y="{y + 17}">{_escape_xml(node["label"])}</text>')
    for link in links:
        start_y = family_y.get(link["source"], 0) + 11
        end_y = mode_y.get(link["target"], 0) + 13
        width_hint = max(1, min(12, int(link["value"])))
        svg_lines.append(
            f'<path class="flow" d="M {left_x + 240} {start_y} C 480 {start_y}, 620 {end_y}, {right_x} {end_y}" stroke-width="{width_hint}"/>'
        )
    svg_lines.append("</svg>")
    return "\n".join(svg_lines)


def render_reasoning_html(trace: dict[str, Any]) -> str:
    svg = render_reasoning_svg(trace)
    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Byte Reasoning Graph</title>"
        "<style>body{font-family:Georgia,serif;background:#f5f8fb;color:#102030;padding:24px}figure{margin:0}</style>"
        "</head><body><h1>Byte Reasoning Graph</h1><figure>"
        + svg
        + "</figure></body></html>"
    )


def _format_ci(payload: Any) -> str:
    data = dict(payload or {})
    return f"{float(data.get('low', 0.0) or 0.0):.4f}-{float(data.get('high', 0.0) or 0.0):.4f}"


def _iter_phases(
    results: dict[str, Any],
) -> Iterable[tuple[str, str, str, dict[str, Any]]]:
    """Yield provider/system/phase payloads from the nested benchmark result tree."""

    for provider_name, provider_payload in (results.get("providers", {}) or {}).items():
        systems = dict(provider_payload.get("systems", {}) or {})
        for system_name, system_payload in systems.items():
            phases = dict(system_payload.get("phases", {}) or {})
            for phase_name, phase_payload in phases.items():
                yield provider_name, system_name, phase_name, dict(phase_payload or {})


def _flatten_phase_data(results: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Flatten the nested provider/system/phase tree into compact summaries and raw records."""

    phase_summaries: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for provider_name, system_name, phase_name, phase_payload in _iter_phases(results):
        phase_summaries.append(
            {
                "provider": provider_name,
                "system": system_name,
                "phase": phase_name,
                "summary": dict(phase_payload.get("summary", {}) or {}),
                "comparison_to_direct": dict(phase_payload.get("comparison_to_direct", {}) or {}),
                "replicate_stats": dict(phase_payload.get("replicate_stats", {}) or {}),
            }
        )
        records.extend(dict(record) for record in (phase_payload.get("records", []) or []))
    return phase_summaries, records


def _has_nested_provider_tree(results: dict[str, Any]) -> bool:
    """Return whether a report payload still carries the verbose provider/system tree."""

    providers = results.get("providers")
    if not isinstance(providers, dict):
        return False
    return any(isinstance(payload, dict) and "systems" in payload for payload in providers.values())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable UTF-8 JSON with a trailing newline."""

    _write_text(path, json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def _write_text(path: Path, text: str) -> None:
    """Write stable UTF-8 text with LF line endings."""

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _preferred_phase(provider_payload: dict[str, Any], system_name: str) -> dict[str, Any] | None:
    phases = (
        ((provider_payload.get("systems", {}) or {}).get(system_name, {}) or {}).get("phases", {})
        or {}
    )
    for candidate in ("warm_100", "warm_1000", "cold"):
        if candidate in phases:
            return phases[candidate]
    return None


def _preferred_phase_name(provider_payload: dict[str, Any], system_name: str) -> str:
    phases = (
        ((provider_payload.get("systems", {}) or {}).get(system_name, {}) or {}).get("phases", {})
        or {}
    )
    for candidate in ("warm_100", "warm_1000", "cold"):
        if candidate in phases:
            return candidate
    return ""


def _ensure_node(nodes: dict[str, dict[str, Any]], node_id: str, label: str) -> None:
    if node_id not in nodes:
        nodes[node_id] = {"id": node_id, "label": label}


def _escape_xml(value: str) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
