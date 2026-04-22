import importlib
from types import ModuleType
from typing import Any


class LazyImport(ModuleType):
    """
    Lazily import a module.
    """

    def __init__(self, local_name, parent_module_globals, name) -> None:
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        super().__init__(name)

    def _load(self) -> Any:
        module = importlib.import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item) -> Any:
        module = self._load()
        return getattr(module, item)

    def __dir__(self) -> Any:
        module = self._load()
        return dir(module)
