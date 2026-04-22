import os
import shutil
import subprocess
from pathlib import Path

import pytest

INTEGRATION_ROOT = Path(__file__).resolve().parent
COMPOSE_FILE = INTEGRATION_ROOT / "docker-compose.yml"
COMPOSE_SERVICES = ["redis-stack", "etcd", "minio", "milvus-standalone"]


def pytest_addoption(parser) -> None:
    group = parser.getgroup("byte-integration")
    group.addoption(
        "--run-live-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require live external services.",
    )
    group.addoption(
        "--integration-stack",
        action="store",
        default="external",
        choices=("external", "docker"),
        help="How to satisfy live integration services.",
    )


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "integration_live: requires external services and is skipped unless --run-live-integration is set",
    )
    config.addinivalue_line(
        "markers",
        "integration_mocked: exercises service-like integration flows against in-process mocks",
    )


def pytest_collection_modifyitems(config, items) -> None:
    if config.getoption("--run-live-integration"):
        return
    skip_live = pytest.mark.skip(reason="live integration tests require --run-live-integration")
    for item in items:
        if "integration_live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(scope="session")
def live_service_stack(request) -> object:
    if not request.config.getoption("--run-live-integration"):
        pytest.skip("live integration tests require --run-live-integration")

    stack_mode = request.config.getoption("--integration-stack")
    if stack_mode != "docker":
        yield {"mode": "external"}
        return

    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker is required for --integration-stack=docker")

    compose_cmd = [docker, "compose", "-f", str(COMPOSE_FILE)]
    env = os.environ.copy()
    env.setdefault("COMPOSE_PROJECT_NAME", "byteai-tests")
    subprocess.run(
        compose_cmd + ["up", "-d"] + COMPOSE_SERVICES,
        check=True,
        env=env,
        cwd=str(INTEGRATION_ROOT),
    )
    try:
        yield {"mode": "docker", "compose_file": str(COMPOSE_FILE)}
    finally:
        subprocess.run(
            compose_cmd + ["down", "-v"],
            check=False,
            env=env,
            cwd=str(INTEGRATION_ROOT),
        )
