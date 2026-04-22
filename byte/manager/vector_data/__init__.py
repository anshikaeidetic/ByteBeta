from typing import Any

__all__ = ["VectorBase"]

from byte.utils.lazy_import import LazyImport

vector_manager = LazyImport("vector_manager", globals(), "byte.manager.vector_data.manager")


def VectorBase(name: str, **kwargs) -> Any:
    """Generate specific VectorBase with the configuration."""
    return vector_manager.VectorBase.get(name, **kwargs)
