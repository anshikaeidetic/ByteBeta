import uuid
from collections.abc import Callable
from typing import Any

from byte import cache
from byte.manager.data_manager import DataManager
from byte.processor.check_hit import check_hit_session
from byte.utils.log import byte_log


class Session:
    """Session-scoped cache isolation and hit filtering."""

    def __init__(
        self,
        name: str | None = None,
        data_manager: DataManager | None = None,
        check_hit_func: Callable | None = None,
    ) -> None:
        self._name = uuid.uuid4().hex if not name else name
        self._data_manager = cache.data_manager if not data_manager else data_manager
        self.check_hit_func = check_hit_session if not check_hit_func else check_hit_func

    @property
    def name(self) -> Any:
        return self._name

    def __enter__(self) -> Any:
        byte_log.warning("The `with` method will delete the session data directly on exit.")
        return self

    def __exit__(self, *_) -> None:
        self.drop()

    def drop(self) -> None:
        """Drop the session and delete all data in the session"""
        self._data_manager.delete_session(self.name)
        byte_log.info("Deleting data in the session: %s.", self.name)
