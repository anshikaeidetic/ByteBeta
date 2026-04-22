"""SQLite-backed worker and replay state for the Byte control plane."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from byte_server._control_plane_store_events import _ControlPlaneStoreEventMixin
from byte_server._control_plane_store_schema import _ControlPlaneStoreSchemaMixin
from byte_server._control_plane_store_settings import _ControlPlaneStoreSettingsMixin
from byte_server._control_plane_store_workers import _ControlPlaneStoreWorkerMixin


class ControlPlaneStore(
    _ControlPlaneStoreSchemaMixin,
    _ControlPlaneStoreSettingsMixin,
    _ControlPlaneStoreWorkerMixin,
    _ControlPlaneStoreEventMixin,
):
    def __init__(self, db_path: str = "byte_control_plane.sqlite3") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._workers: dict[str, dict] = {}
        self._kv_residency: dict[str, dict] = {}
        self._recommendations: dict[str, dict] = {}
        self._ensure_schema()
        self._load_mirrors()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> Any:
        conn = sqlite3.connect(self._db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn
