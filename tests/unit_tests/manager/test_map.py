import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from byte.manager.data_manager import MapDataManager
from byte.utils.error import CacheError

data_map_path = "data_map.txt"


class UnsafePayload:
    pass


def test_map() -> None:
    if os.path.isfile(data_map_path):
        os.remove(data_map_path)

    data_manager = MapDataManager(data_map_path, 3)
    a = "a"
    for i in range(4):
        data_manager.save(chr(ord(a) + i), str(i), chr(ord(a) + i))
    assert len(data_manager.search("a")) == 0
    question, answer, emb, _, _ = data_manager.search("b")[0]
    assert question == "b", question
    assert answer == "1", answer
    assert emb == "b", emb
    data_manager.close()


def test_map_round_trips_persisted_entries(tmp_path) -> None:
    data_path = tmp_path / "data_map.pkl"
    data_manager = MapDataManager(str(data_path), 3)
    data_manager.save("hello", "world", "hello")
    data_manager.close()

    reloaded = MapDataManager(str(data_path), 3)
    question, answer, emb, _, _ = reloaded.search("hello")[0]
    assert question == "hello"
    assert answer == "world"
    assert emb == "hello"


def test_map_import_data_uses_hashable_embedding_keys(tmp_path) -> None:
    data_manager = MapDataManager(str(tmp_path / "data_map.pkl"), 3)
    embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    data_manager.import_data(
        questions=["q1"],
        answers=["a1"],
        embedding_datas=[embedding],
        session_ids=[None],
    )

    question, answer, emb, _, _ = data_manager.search(np.array([1.0, 2.0, 3.0], dtype=np.float32))[
        0
    ]
    assert question == "q1"
    assert answer == "a1"
    assert np.array_equal(emb, embedding)


def test_map_rejects_unsafe_pickle_payload(tmp_path) -> None:
    data_path = tmp_path / "unsafe.pkl"
    with data_path.open("wb") as handle:
        pickle.dump(UnsafePayload(), handle)

    with pytest.raises(CacheError, match="unsafe cache payload class"):
        MapDataManager(str(data_path), 3)


def test_map_flush_is_safe_during_parallel_saves(tmp_path) -> None:
    data_manager = MapDataManager(str(tmp_path / "concurrent_map.pkl"), 256)

    def writer(offset) -> None:
        for index in range(40):
            key = f"{offset}-{index}"
            data_manager.save(f"q-{key}", f"a-{key}", key)

    def flusher() -> None:
        for _ in range(20):
            data_manager.flush()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(writer, "left"),
            pool.submit(writer, "right"),
            pool.submit(flusher),
            pool.submit(flusher),
        ]
        for future in futures:
            future.result()

    assert data_manager.search("left-0")
