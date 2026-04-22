"""Helpers for loading optional dependencies only when a feature is used."""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType
from typing import Any

from byte._optional_features import missing_dependency_error_for_module


def _missing_library_error(libname: str, package: str | None = None) -> ModuleNotFoundError:
    return missing_dependency_error_for_module(libname, package=package)


def _check_library(libname: str, prompt: bool | None = None, package: str | None = None) -> bool:
    del prompt
    if importlib.util.find_spec(libname):
        return True
    raise _missing_library_error(libname, package)


def load_optional_module(module_name: str, package: str | None = None) -> ModuleType:
    """Import an optional module on demand with Byte's install guidance."""

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or module_name.startswith(f"{exc.name}."):
            raise _missing_library_error(module_name, package) from exc
        raise


def load_optional_attr(module_name: str, attr_name: str, package: str | None = None) -> Any:
    """Load an attribute from an optional module on demand."""

    return getattr(load_optional_module(module_name, package=package), attr_name)


class LazyOptionalModule:
    """Proxy module object that resolves an optional dependency only on first use."""

    def __init__(self, module_name: str, package: str | None = None) -> None:
        self._module_name = module_name
        self._package = package
        self._module: ModuleType | None = None

    def _load(self) -> ModuleType:
        if self._module is None:
            self._module = load_optional_module(self._module_name, package=self._package)
        return self._module

    def __getattr__(self, item: str) -> Any:
        return getattr(self._load(), item)

    def __dir__(self) -> list[str]:
        return dir(self._load())


def lazy_optional_module(module_name: str, package: str | None = None) -> LazyOptionalModule:
    """Create a lazy proxy for an optional dependency module."""

    return LazyOptionalModule(module_name, package=package)


class LazyOptionalAttr:
    """Proxy an attribute from an optional module without importing it eagerly."""

    def __init__(self, module_name: str, attr_name: str, package: str | None = None) -> None:
        object.__setattr__(self, "_module_name", module_name)
        object.__setattr__(self, "_attr_name", attr_name)
        object.__setattr__(self, "_package", package)
        object.__setattr__(self, "_overrides", {})

    def _load(self) -> Any:
        return load_optional_attr(
            self._module_name,
            self._attr_name,
            package=self._package,
        )

    def _resolve_member(self, item: str) -> Any:
        overrides = object.__getattribute__(self, "_overrides")
        if item in overrides:
            return overrides[item]
        return getattr(self._load(), item)

    def __getattr__(self, item: str) -> Any:
        overrides = object.__getattribute__(self, "_overrides")
        if item in overrides:
            return overrides[item]
        return _LazyOptionalMember(self, item)

    def __setattr__(self, key: str, value) -> None:
        if key.startswith("_"):
            object.__setattr__(self, key, value)
            return
        overrides = object.__getattribute__(self, "_overrides")
        overrides[key] = value

    def __call__(self, *args, **kwargs) -> Any:
        return self._load()(*args, **kwargs)

    def __dir__(self) -> list[str]:
        return dir(self._load())


def lazy_optional_attr(
    module_name: str,
    attr_name: str,
    package: str | None = None,
) -> LazyOptionalAttr:
    """Create a lazy proxy for an attribute from an optional dependency module."""

    return LazyOptionalAttr(module_name, attr_name, package=package)


class _LazyOptionalMember:
    """Proxy a member on a lazy optional attribute."""

    def __init__(self, parent: LazyOptionalAttr, name: str) -> None:
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_name", name)

    def _load(self) -> Any:
        parent = object.__getattribute__(self, "_parent")
        name = object.__getattribute__(self, "_name")
        return parent._resolve_member(name)

    def __call__(self, *args, **kwargs) -> Any:
        return self._load()(*args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._load(), item)

    def __dir__(self) -> list[str]:
        return dir(self._load())
