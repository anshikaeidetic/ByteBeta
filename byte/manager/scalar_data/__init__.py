from typing import Any

__all__ = ["CacheBase"]

from byte.utils.lazy_import import LazyImport

scalar_manager = LazyImport("scalar_manager", globals(), "byte.manager.scalar_data.manager")


def CacheBase(name: str, **kwargs) -> Any:
    """Generate specific CacheStorage with the configuration. For example, setting for
    `SQLDataBase` (with `name`, `sql_url` and `table_name` params) to manage SQLite, PostgreSQL, MySQL, MariaDB, SQL Server and Oracle.
    """
    return scalar_manager.CacheBase.get(name, **kwargs)
