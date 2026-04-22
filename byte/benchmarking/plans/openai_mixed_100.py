"""Package entrypoint for the 100-request OpenAI mixed workload plan."""

from byte.benchmarking.programs import deep_openai_100_request_mixed_benchmark as _script

build_workload_plan = _script.build_workload_plan
provider_coverage = _script.provider_coverage

__all__ = ["build_workload_plan", "provider_coverage"]
