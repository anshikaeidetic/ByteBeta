import logging
import os
import uuid
from pathlib import Path
from typing import Any

from byte.manager.object_data.base import ObjectBase
from byte.utils.error import ByteErrorCode
from byte.utils.log import byte_log, log_byte_error


class LocalObjectStorage(ObjectBase):
    """Local object storage"""

    def __init__(self, local_root: str) -> None:
        self._local_root = Path(local_root).expanduser().resolve()
        self._local_root.mkdir(parents=True, exist_ok=True)

    def _resolve_managed_path(self, obj: str) -> Any | None:
        try:
            candidate = Path(obj).expanduser().resolve()
            candidate.relative_to(self._local_root)
            return candidate
        except (OSError, RuntimeError, ValueError):
            return None

    def put(self, obj: Any) -> str:
        f_path = self._local_root / str(uuid.uuid4())
        with open(f_path, "wb") as f:
            f.write(obj)
        return str(f_path.absolute())

    def get(self, obj: str) -> Any:
        target = self._resolve_managed_path(obj)
        if target is None:
            return None
        try:
            with open(target, "rb") as f:
                return f.read()
        except OSError:
            return None

    def get_access_link(self, obj: str, _: int = 3600) -> Any:
        return obj

    def delete(self, to_delete: list[str]) -> None:
        assert isinstance(to_delete, list)
        for obj in to_delete:
            target = self._resolve_managed_path(obj)
            if target is None:
                byte_log.warning("Refusing to delete object outside local object root: %s", obj)
                continue
            try:
                os.remove(target)
            except FileNotFoundError:
                byte_log.warning("Can not find obj: %s", obj)
            except OSError as exc:
                log_byte_error(
                    byte_log,
                    logging.WARNING,
                    "Failed deleting local object payload.",
                    error=exc,
                    code=ByteErrorCode.STORAGE_WRITE,
                    boundary="storage.write",
                    stage="local_object_delete",
                )
