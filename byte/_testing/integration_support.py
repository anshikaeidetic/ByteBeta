"""Typed helpers shared by Byte's integration-style tests.

These helpers live under the importable ``byte`` package so pytest's
``--import-mode=importlib`` can collect tests without depending on
repo-root path mutation or ad hoc sibling imports from ``tests/``.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

DEFAULT_TEST_ARTIFACTS = ("sqlite.db", "faiss.index")
_LOGGER_NAME = "byte.tests.integration"
_STREAM_FORMAT = "[%(asctime)s - %(levelname)s - %(name)s]: %(message)s (%(filename)s:%(lineno)s)"


def _log_directory() -> Path:
    configured = os.environ.get("CI_LOG_PATH")
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / "byte-ci-logs"


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_file_handler(path: Path, level: int) -> logging.Handler | None:
    try:
        handler: logging.Handler = logging.FileHandler(path, encoding="utf-8")
    except OSError:
        return None
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_STREAM_FORMAT))
    return handler


def _configure_test_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    log_dir = _ensure_directory(_log_directory())
    file_targets = (
        (log_dir / "ci_test_log.debug", logging.DEBUG),
        (log_dir / "ci_test_log.log", logging.INFO),
        (log_dir / "ci_test_log.err", logging.ERROR),
    )
    worker_name = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_name:
        file_targets += ((log_dir / f"{worker_name}.log", logging.DEBUG),)

    for file_path, level in file_targets:
        handler = _build_file_handler(file_path, level)
        if handler is not None:
            logger.addHandler(handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(logging.Formatter(_STREAM_FORMAT))
    logger.addHandler(stream_handler)
    return logger


test_log = _configure_test_logger()


def remove_test_artifacts(file_names: Iterable[str] | None = None) -> None:
    """Delete integration-test artifacts from the working directory."""

    for file_name in file_names or DEFAULT_TEST_ARTIFACTS:
        path = Path(file_name)
        if path.is_file():
            path.unlink()
            test_log.info("%s is removed", path)


def log_time_func(func_name: str, delta_time: float) -> None:
    """Record benchmark-style timing output for legacy integration tests."""

    test_log.info("func `%s` consume time: %.2fs", func_name, delta_time)


def disable_cache(*_: object, **__: object) -> bool:
    """Legacy hook that forces cache bypass in integration tests."""

    return False


class IntegrationTestBase:
    """Common setup and teardown for Byte integration tests."""

    def setup_method(self, method: object) -> None:
        method_name = getattr(method, "__name__", repr(method))
        test_log.info("%s setup %s", "*" * 35, "*" * 35)
        test_log.info("[setup_method] Start setup test case %s.", method_name)
        test_log.info("[setup_method] Clean up tmp files.")
        remove_test_artifacts()

    def teardown_method(self, method: object) -> None:
        method_name = getattr(method, "__name__", repr(method))
        test_log.info("%s teardown %s", "*" * 35, "*" * 35)
        test_log.info("[teardown_method] Start teardown test case %s...", method_name)
        test_log.info("[teardown_method] Clean up tmp files.")
        remove_test_artifacts()


__all__ = [
    "IntegrationTestBase",
    "disable_cache",
    "log_time_func",
    "remove_test_artifacts",
    "test_log",
]
