from typing import Any

__all__ = ["ObjectBase"]

from byte.utils.lazy_import import LazyImport

object_manager = LazyImport("object_manager", globals(), "byte.manager.object_data.manager")


def ObjectBase(name: str, **kwargs) -> Any:
    """Generate specific ObjectStorage with the configuration. For example, setting for
    `ObjectBase` (with `name`) to manage LocalObjectStorage, S3 object storage.
    """
    return object_manager.ObjectBase.get(name, **kwargs)
