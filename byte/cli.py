"""ByteAI developer CLI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from byte.benchmarking.quickstart import run_product_benchmark

try:  # pragma: no cover - Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore


_DEFAULT_TOML = """[gateway]
host = "127.0.0.1"
port = 8000
cache_dir = "byte_data"
cache_mode = "exact"
gateway_mode = "backend"

[control_plane]
db_path = "byte_control_plane.db"
replay_enabled = false
replay_sample_rate = 0.05
worker_urls = []
memory_service_url = ""
"""


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _server_command_from_config(config: dict[str, Any]) -> list[str]:
    gateway = dict(config.get("gateway", {}) or {})
    control_plane = dict(config.get("control_plane", {}) or {})
    command = [
        sys.executable,
        "-m",
        "byte_server.server",
        "--gateway",
        "True",
        "--gateway-cache-mode",
        str(gateway.get("cache_mode", "exact") or "exact"),
        "--gateway-mode",
        str(gateway.get("gateway_mode", "backend") or "backend"),
        "--host",
        str(gateway.get("host", "127.0.0.1") or "127.0.0.1"),
        "--port",
        str(int(gateway.get("port", 8000) or 8000)),
        "--cache-dir",
        str(gateway.get("cache_dir", "byte_data") or "byte_data"),
        "--control-plane-db",
        str(control_plane.get("db_path", "byte_control_plane.db") or "byte_control_plane.db"),
    ]
    for worker_url in control_plane.get("worker_urls", []) or []:
        command.extend(["--control-plane-worker-url", str(worker_url)])
    memory_service_url = str(control_plane.get("memory_service_url", "") or "").strip()
    if memory_service_url:
        command.extend(["--memory-service-url", memory_service_url])
    if bool(control_plane.get("replay_enabled", False)):
        command.append("--replay-enabled")
    replay_rate = float(control_plane.get("replay_sample_rate", 0.05) or 0.05)
    command.extend(["--replay-sample-rate", str(replay_rate)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(prog="byte")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--path", default="byteai.toml")
    init_parser.add_argument("--force", action="store_true")

    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--config", default="byteai.toml")

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--provider", default="openai")
    benchmark_parser.add_argument("--profile", default="tier1")
    benchmark_parser.add_argument("--systems", default="")
    benchmark_parser.add_argument("--phases", default="")
    benchmark_parser.add_argument("--out-dir", default="artifacts/benchmarks")
    benchmark_parser.add_argument("--live", default="auto", choices=["auto", "on", "off"])
    benchmark_parser.add_argument("--compare-baseline", action="store_true")
    benchmark_parser.add_argument("--max-items-per-family", type=int, default=3)
    benchmark_parser.add_argument("--concurrency", type=int, default=4)

    args = parser.parse_args()
    if args.command == "init":
        target = Path(str(args.path or "byteai.toml")).expanduser()
        if target.exists() and not args.force:
            raise SystemExit(f"{target} already exists. Use --force to overwrite it.")
        target.write_text(_DEFAULT_TOML, encoding="utf-8")
        print(f"Wrote {target}")
        return

    if args.command == "start":
        config_path = Path(str(args.config or "byteai.toml")).expanduser()
        config = _load_toml(config_path)
        if not config:
            config = tomllib.loads(_DEFAULT_TOML)
        command = _server_command_from_config(config)
        raise SystemExit(subprocess.call(command))

    if args.command == "benchmark":
        systems = [part.strip() for part in str(args.systems or "").split(",") if part.strip()]
        phases = [part.strip() for part in str(args.phases or "").split(",") if part.strip()]
        run_product_benchmark(
            provider=str(args.provider),
            profile=str(args.profile),
            systems=systems or None,
            phases=phases or None,
            out_dir=str(args.out_dir),
            live=str(args.live),
            compare_baseline=bool(args.compare_baseline),
            max_items_per_family=int(args.max_items_per_family),
            concurrency=int(args.concurrency),
        )
        return


__all__ = ["main"]


if __name__ == "__main__":
    main()
