import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from byte.benchmarking.runner import run_suite
from byte.benchmarking.workload_generator import write_workloads


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Byte production benchmark suite.")
    parser.add_argument("--profile", default="tier1_v2_deepseek")
    parser.add_argument("--providers", default="deepseek")
    parser.add_argument("--systems", default="direct,native_cache,byte")
    parser.add_argument("--phase", default="cold,warm_100,warm_1000")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-items-per-family", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--scorecard", default="dual", choices=["dual", "forced", "selective"])
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--judge-mode", default="disabled", choices=["disabled", "appendix", "panel"])
    parser.add_argument("--contamination-check", action="store_true", default=False)
    parser.add_argument("--live-cutoff-date", default="")
    parser.add_argument("--release-gate", action="store_true", default=False)
    parser.add_argument("--fail-on-thresholds", action="store_true", default=True)
    parser.add_argument("--no-fail-on-thresholds", dest="fail_on_thresholds", action="store_false")
    args = parser.parse_args()

    write_workloads()
    providers = [item.strip() for item in str(args.providers or "").split(",") if item.strip()]
    systems = [item.strip() for item in str(args.systems or "").split(",") if item.strip()]
    phases = [item.strip() for item in str(args.phase or "").split(",") if item.strip()]
    results = run_suite(
        profile=str(args.profile or "tier1"),
        providers=providers,
        systems=systems,
        phases=phases,
        out_dir=str(args.out_dir),
        max_items_per_family=args.max_items_per_family,
        concurrency=args.concurrency,
        fail_on_thresholds=bool(args.fail_on_thresholds),
        scorecard_mode=str(args.scorecard or "dual"),
        replicates=max(1, int(args.replicates or 1)),
        confidence_level=float(args.confidence_level or 0.95),
        judge_mode=str(args.judge_mode or "disabled"),
        contamination_check=bool(args.contamination_check),
        live_cutoff_date=str(args.live_cutoff_date or ""),
        release_gate=bool(args.release_gate),
    )
    report_path = Path(results["artifacts"]["engineering_json"])
    print(f"Benchmark suite complete. Engineering report: {report_path}")
    print(json.dumps({"failed_thresholds": results["failed_thresholds"], "artifacts": results["artifacts"]}, indent=2))
    return 2 if results["failed_thresholds"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
