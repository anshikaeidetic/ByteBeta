"""Utilities for backend adapter compatibility wrappers."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any


class _BackendProxyModule(ModuleType):
    """Forward wrapper attribute access and patching to the backend module."""

    _byte_backend: ModuleType

    def __getattr__(self, name: str) -> Any:
        return getattr(self._byte_backend, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("__") or name in {"_byte_backend", "__all__"}:
            super().__setattr__(name, value)
            return
        setattr(self._byte_backend, name, value)
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name.startswith("__") or name in {"_byte_backend", "__all__"}:
            super().__delattr__(name)
            return
        if hasattr(self._byte_backend, name):
            delattr(self._byte_backend, name)
        if name in self.__dict__:
            super().__delattr__(name)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(dir(self._byte_backend)))


def bind_backend_module(module_name: str, backend: ModuleType) -> list[str]:
    """Bind a thin wrapper module to a backend without aliasing module objects."""

    module = sys.modules[module_name]
    module.__class__ = _BackendProxyModule
    module._byte_backend = backend

    explicit_public_names = list(module.__dict__.get("__all__", []))
    public_names = explicit_public_names or list(getattr(backend, "__all__", [])) or [
        name for name in dir(backend) if not name.startswith("_")
    ]
    module.__all__ = public_names
    return public_names
