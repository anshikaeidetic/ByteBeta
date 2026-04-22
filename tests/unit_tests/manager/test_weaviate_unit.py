from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np

from byte.manager.vector_data import weaviate as weaviate_module
from byte.manager.vector_data.base import VectorData
from byte.manager.vector_data.weaviate import Weaviate


class _FakeBatch:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.kwargs: dict[str, object] | None = None

    def __call__(self, **kwargs) -> object:
        self.kwargs = kwargs
        return self

    def __enter__(self) -> object:
        return self

    def __exit__(self, exc_type, exc, tb) -> object:
        del exc_type, exc, tb
        return False

    def add_data_object(self, *, data_object, class_name, vector) -> None:
        self.calls.append(
            {
                "data_object": data_object,
                "class_name": class_name,
                "vector": vector,
            }
        )

    def flush(self) -> None:
        return None


class _FakeQuery:
    def __init__(self, client) -> None:
        self.client = client
        self.payload = None

    def get(self, class_name, properties) -> object:
        self.payload = {
            "class_name": class_name,
            "properties": properties,
            "with_where": None,
            "with_additional": None,
            "with_limit": None,
            "with_near_vector": None,
        }
        return self

    def with_near_vector(self, content) -> object:
        self.payload["with_near_vector"] = content
        return self

    def with_additional(self, fields) -> object:
        self.payload["with_additional"] = fields
        return self

    def with_limit(self, limit) -> object:
        self.payload["with_limit"] = limit
        return self

    def with_where(self, where) -> object:
        self.payload["with_where"] = where
        return self

    def do(self) -> object:
        return self.client.query_results.pop(0)


class _FakeDataObject:
    def __init__(self) -> None:
        self.deleted: list[dict[str, object]] = []
        self.created: list[dict[str, object]] = []

    def delete(self, *, class_name, uuid) -> None:
        self.deleted.append({"class_name": class_name, "uuid": uuid})

    def create(self, *, data_object, class_name, vector) -> None:
        self.created.append(
            {
                "data_object": data_object,
                "class_name": class_name,
                "vector": vector,
            }
        )


class _FakeSchema:
    def __init__(self, exists: bool) -> None:
        self._exists = exists
        self.created: list[dict[str, object]] = []

    def exists(self, class_name) -> object:
        return self._exists

    def create_class(self, class_schema) -> None:
        self.created.append(class_schema)


class _FakeClient:
    def __init__(self, exists: bool = False) -> None:
        self.schema = _FakeSchema(exists=exists)
        self.batch = _FakeBatch()
        self.query_results: list[dict[str, object]] = []
        self.query = _FakeQuery(self)
        self.data_object = _FakeDataObject()


def test_weaviate_creates_default_schema_and_supports_vector_operations() -> None:
    fake_client = _FakeClient(exists=False)
    fake_client.query_results = [
        {
            "data": {
                "Get": {
                    "byte": [
                        {"_additional": {"distance": 0.1}, "data_id": 1},
                        {"_additional": {"distance": 0.2}, "data_id": 2},
                    ]
                }
            }
        },
        {"data": {"Get": {"byte": [{"_additional": {"id": "uuid-1"}}]}}},
        {
            "data": {
                "Get": {
                    "byte": [
                        {"_additional": {"vector": [1.0, 2.0]}, "data_id": 6},
                    ]
                }
            }
        },
        {"data": {"Get": {"byte": []}}},
        {"data": {"Get": {"byte": [{"_additional": {"id": "uuid-6"}}]}}},
    ]

    fake_weaviate = SimpleNamespace(Client=mock.Mock(return_value=fake_client))
    fake_embedded = SimpleNamespace(EmbeddedOptions=mock.Mock(return_value="embedded"))
    with mock.patch.object(weaviate_module, "weaviate", fake_weaviate):
        with mock.patch.object(weaviate_module, "weaviate_embedded", fake_embedded):
            store = Weaviate()

    assert store.class_name == "byte"
    assert fake_client.schema.created[0]["class"] == "byte"

    vectors = [
        VectorData(id=1, data=np.array([1.0, 0.0], dtype="float32")),
        VectorData(id=2, data=np.array([0.0, 1.0], dtype="float32")),
    ]
    store.mul_add(vectors)
    assert fake_client.batch.kwargs == {"batch_size": 100, "dynamic": True}
    assert len(fake_client.batch.calls) == 2
    assert store.search(np.array([1.0, 0.0], dtype="float32"), top_k=2) == [(0.1, 1), (0.2, 2)]

    store.delete([1])
    assert fake_client.data_object.deleted == [{"class_name": "byte", "uuid": "uuid-1"}]

    emb = store.get_embeddings(6)
    assert np.array_equal(emb, np.array([1.0, 2.0], dtype="float32"))
    assert store.get_embeddings(99) is None

    store.update_embeddings(6, np.array([9.0, 8.0], dtype="float32"))
    assert fake_client.data_object.created[0]["data_object"] == {"data_id": 6}
    assert np.array_equal(fake_client.data_object.created[0]["vector"], np.array([9.0, 8.0], dtype="float32"))

    store.close()


def test_weaviate_reuses_existing_class_schema_without_recreating() -> None:
    fake_client = _FakeClient(exists=True)
    class_schema = {
        "class": "CustomCache",
        "description": "LLM response cache",
        "properties": [{"name": "data_id", "dataType": ["int"]}],
        "vectorIndexConfig": {"distance": "cosine"},
    }

    fake_weaviate = SimpleNamespace(Client=mock.Mock(return_value=fake_client))
    with mock.patch.object(weaviate_module, "weaviate", fake_weaviate):
        store = Weaviate(url="http://localhost:8080", class_schema=class_schema)

    assert store.class_name == "CustomCache"
    assert fake_client.schema.created == []
