import unittest

from byte.manager import CacheBase, VectorBase
from byte.manager.scalar_data.manager import CacheBase as InnerCacheBase
from byte.manager.vector_data.manager import VectorBase as InnerVectorBase
from byte.utils.error import NotFoundError


class TestBaseStore(unittest.TestCase):
    def test_cache_base(self) -> None:
        with self.assertRaises(EnvironmentError):
            InnerCacheBase()

        with self.assertRaises(NotFoundError):
            CacheBase("test_cache_base")

    def test_vector_base(self) -> None:
        with self.assertRaises(EnvironmentError):
            InnerVectorBase()

        with self.assertRaises(NotFoundError):
            VectorBase("test_cache_base")
