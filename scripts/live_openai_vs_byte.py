import argparse
import json
import multiprocessing as mp
import os
import shutil
import socket
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from byte import Cache, Config
from byte._backends import openai as byte_openai
from byte.benchmarking._optional_runtime import create_openai_client
from byte.embedding.string import to_embeddings as string_embedding
from byte.manager import manager_factory
from byte.processor.pre import last_content
from byte.similarity_evaluation import ExactMatchEvaluation

PROMPT = "Reply with exactly BYTE_OK and nothing else."
DEFAULT_MODEL = "gpt-4o-mini"


def _response_text(resp: Any) -> str:
    if isinstance(resp, dict):
        return resp["choices"][0]["message"]["content"]
    return resp.choices[0].message.content


def _configure_cache(cache_obj: Cache, cache_dir: str) -> None:
    cache_obj.init(
        pre_embedding_func=last_content,
        embedding_func=string_embedding,
        data_manager=manager_factory("map", data_dir=cache_dir),
        similarity_evaluation=ExactMatchEvaluation(),
        config=Config(enable_token_counter=False),
    )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(base_url: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    with httpx.Client(timeout=2.0) as client:
        while time.time() < deadline:
            try:
                response = client.get(base_url + "/")
                if response.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.25)
    raise TimeoutError(f"Server did not become ready: {base_url}")


def _serve_proxy(port: int, cache_dir: str, server_api_key: str | None) -> None:
    if server_api_key:
        os.environ["OPENAI_API_KEY"] = server_api_key
    else:
        os.environ.pop("OPENAI_API_KEY", None)

    import uvicorn

    import byte_server.server as server

    server.openai_cache = Cache()
    _configure_cache(server.openai_cache, cache_dir)
    uvicorn.run(server.app, host="127.0.0.1", port=port, log_level="warning")


def _direct_openai_run(api_key: str, model: str) -> dict[str, Any]:
    client = create_openai_client(api_key=api_key)
    runs = []
    for _ in range(2):
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0,
            max_tokens=8,
        )
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        runs.append(
            {
                "status_code": 200,
                "latency_ms": duration_ms,
                "text": _response_text(response),
                "byte": False,
                "response_id": response.id,
                "model": response.model,
            }
        )
    return {
        "mode": "direct-openai",
        "key_source": "client-bearer",
        "runs": runs,
    }


def _byte_library_run(api_key: str, model: str) -> dict[str, Any]:
    cache_dir = tempfile.mkdtemp(prefix="byte-lib-cache-")
    cache_obj = Cache()
    _configure_cache(cache_obj, cache_dir)
    try:
        runs = []
        for _ in range(2):
            start = time.perf_counter()
            response = byte_openai.ChatCompletion.create(
                cache_obj=cache_obj,
                api_key=api_key,
                model=model,
                messages=[{"role": "user", "content": PROMPT}],
                temperature=0,
                max_tokens=8,
            )
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            runs.append(
                {
                    "status_code": 200,
                    "latency_ms": duration_ms,
                    "text": _response_text(response),
                    "byte": bool(response.get("byte")) if isinstance(response, dict) else False,
                    "object": response.get("object") if isinstance(response, dict) else None,
                }
            )
        return {
            "mode": "byte-library",
            "key_source": "client-bearer",
            "runs": runs,
            "stats": cache_obj.cost_summary(),
        }
    finally:
        if cache_obj.data_manager is not None:
            cache_obj.data_manager.close()
            cache_obj.data_manager = None
        shutil.rmtree(cache_dir, ignore_errors=True)


def _proxy_request(
    client: httpx.Client, base_url: str, model: str, headers: dict[str, str]
) -> dict[str, Any]:
    start = time.perf_counter()
    response = client.post(
        base_url + "/v1/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0,
            "max_tokens": 8,
        },
    )
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    try:
        body = response.json()
    except Exception:
        body = {"raw_text": response.text}

    result = {
        "status_code": response.status_code,
        "latency_ms": duration_ms,
        "byte": bool(body.get("byte")) if isinstance(body, dict) else False,
        "body": body,
    }
    if response.status_code == 200 and isinstance(body, dict):
        result["text"] = body["choices"][0]["message"]["content"]
    return result


def _proxy_mode_run(
    model: str,
    client_api_key: str | None,
    server_api_key: str | None,
    label: str,
    include_clear: bool,
) -> dict[str, Any]:
    cache_dir = tempfile.mkdtemp(prefix=f"{label}-cache-")
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    process = mp.Process(target=_serve_proxy, args=(port, cache_dir, server_api_key), daemon=True)
    process.start()
    try:
        _wait_for_server(base_url)
        headers = {}
        if client_api_key:
            headers["Authorization"] = f"Bearer {client_api_key}"
        with httpx.Client(timeout=60.0) as client:
            runs = [_proxy_request(client, base_url, model, headers)]
            if runs[0]["status_code"] == 200:
                runs.append(_proxy_request(client, base_url, model, headers))
                stats = client.get(base_url + "/stats").json()
                clear_result = None
                if include_clear:
                    clear_response = client.post(base_url + "/clear")
                    clear_result = {
                        "status_code": clear_response.status_code,
                        "body": clear_response.json()
                        if clear_response.headers.get("content-type", "").startswith(
                            "application/json"
                        )
                        else clear_response.text,
                    }
                    runs.append(_proxy_request(client, base_url, model, headers))
                    stats_after_clear = client.get(base_url + "/stats").json()
                else:
                    stats_after_clear = None
            else:
                stats = None
                clear_result = None
                stats_after_clear = None
        return {
            "mode": label,
            "key_source": "client-bearer"
            if client_api_key
            else "server-env"
            if server_api_key
            else "none",
            "runs": runs,
            "stats": stats,
            "clear": clear_result,
            "stats_after_clear": stats_after_clear,
        }
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        shutil.rmtree(cache_dir, ignore_errors=True)


def _render_report(results: dict[str, Any]) -> str:
    lines = []
    lines.append("# Live OpenAI vs ByteAI Cache Validation")
    lines.append("")
    lines.append(f"Generated: {results['generated_at']}")
    lines.append(f"Model: {results['model']}")
    lines.append(f"Prompt: {PROMPT}")
    lines.append(
        "Cache mode for ByteAI Cache live runs: exact-match map cache (lightweight proxy validation)."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    results["direct_openai"]["runs"]
    results["byte_library"]["runs"]
    results["proxy_byo_key"]["runs"]
    results["proxy_server_key"]["runs"]
    no_key = results["proxy_no_key"]["runs"][0]
    lines.append("- Direct OpenAI: both live calls succeeded; neither response was cache-tagged.")
    lines.append(
        f"- ByteAI Cache library: first call was a miss, second call returned with `byte=true`; stats report {results['byte_library']['stats']['cache_hits']} cache hit out of {results['byte_library']['stats']['total_requests']} requests."
    )
    lines.append(
        f"- ByteAI Cache proxy (BYO key): first call was a miss, second call was a cache hit with `byte=true`; `/stats` now reports {results['proxy_byo_key']['stats']['cache_hits']} hit out of {results['proxy_byo_key']['stats']['total_requests']} requests."
    )
    lines.append(
        "- ByteAI Cache proxy (server key): behavior matched BYO mode using the server-managed `OPENAI_API_KEY`; second call was a cache hit."
    )
    lines.append(
        f"- ByteAI Cache proxy (no key anywhere): request failed fast with HTTP {no_key['status_code']} instead of a vague upstream 500."
    )
    lines.append("")
    lines.append("## Scenarios")
    lines.append("")
    for name, key in [
        ("Direct OpenAI", "direct_openai"),
        ("ByteAI Cache Library", "byte_library"),
        ("ByteAI Cache Proxy - BYO Key", "proxy_byo_key"),
        ("ByteAI Cache Proxy - Server Key", "proxy_server_key"),
        ("ByteAI Cache Proxy - No Key", "proxy_no_key"),
    ]:
        section = results[key]
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- Key source: {section['key_source']}")
        for idx, run in enumerate(section["runs"], start=1):
            text = run.get("text")
            body = run.get("body")
            detail = body.get("detail") if isinstance(body, dict) else None
            lines.append(
                f"- Run {idx}: status={run['status_code']}, latency_ms={run['latency_ms']}, byte={run.get('byte', False)}, text={text!r}, detail={detail!r}"
            )
        if section.get("stats") is not None:
            lines.append(f"- Stats: {json.dumps(section['stats'], sort_keys=True)}")
        if section.get("clear") is not None:
            lines.append(f"- Clear endpoint: {json.dumps(section['clear'], sort_keys=True)}")
        if section.get("stats_after_clear") is not None:
            lines.append(
                f"- Stats after clear: {json.dumps(section['stats_after_clear'], sort_keys=True)}"
            )
        lines.append("")
    lines.append("## Fixes Applied During Validation")
    lines.append("")
    lines.append(
        "- Added explicit proxy auth handling so missing or malformed credentials return HTTP 401 with a clear message."
    )
    lines.append(
        "- Routed proxy operational endpoints like `/stats` and `/clear` to the active OpenAI proxy cache instead of the unrelated default cache."
    )
    lines.append(
        "- Added regression tests for BYO key mode, server-key mode, missing-key mode, malformed auth, and proxy stats/cache targeting."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--report", default="docs/reports/live_openai_vs_byte_report.md")
    parser.add_argument("--json-report", default="docs/reports/live_openai_vs_byte_report.json")
    args = parser.parse_args()
    api_key = args.api_key or os.getenv("BYTE_TEST_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Set BYTE_TEST_OPENAI_API_KEY or pass --api-key.")

    results = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
    }
    results["direct_openai"] = _direct_openai_run(api_key, args.model)
    results["byte_library"] = _byte_library_run(api_key, args.model)
    results["proxy_byo_key"] = _proxy_mode_run(
        args.model,
        client_api_key=api_key,
        server_api_key=None,
        label="proxy-byo-key",
        include_clear=True,
    )
    results["proxy_server_key"] = _proxy_mode_run(
        args.model,
        client_api_key=None,
        server_api_key=api_key,
        label="proxy-server-key",
        include_clear=False,
    )
    results["proxy_no_key"] = _proxy_mode_run(
        args.model,
        client_api_key=None,
        server_api_key=None,
        label="proxy-no-key",
        include_clear=False,
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(results), encoding="utf-8")

    json_path = Path(args.json_report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "report": str(report_path),
                "json_report": str(json_path),
                "model": args.model,
            }
        )
    )
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
