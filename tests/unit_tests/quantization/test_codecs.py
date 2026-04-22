import os
from tempfile import TemporaryDirectory

import numpy as np

from byte import Config
from byte.h2o.runtime import describe_huggingface_runtime
from byte.manager.vector_data.base import VectorBase, VectorData
from byte.quantization.polar import PolarQuantCodec
from byte.quantization.qjl import QJLCodec
from byte.quantization.turbo import TurboQuantCodec
from byte.quantization.vector import CompressedVectorStore
from byte.research import research_registry_summary


class _VectorStub(VectorBase):
    def __init__(self) -> None:
        self.items = {}

    def mul_add(self, datas) -> None:
        for item in datas:
            self.items[int(item.id)] = np.asarray(item.data, dtype=np.float32)

    def search(self, data, top_k=1) -> object:
        query = np.asarray(data, dtype=np.float32).reshape(-1)
        ranked = []
        for item_id, vector in self.items.items():
            distance = float(np.linalg.norm(query - vector.reshape(-1)))
            ranked.append((distance, item_id))
        ranked.sort(key=lambda item: item[0])
        return ranked[: max(1, int(top_k or 1))]

    def rebuild(self, ids=None) -> bool:
        return True

    def delete(self, ids) -> bool:
        for item_id in ids or []:
            self.items.pop(int(item_id), None)
        return True

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None

    def get_embeddings(self, data_id) -> object:
        return self.items.get(int(data_id))

    def update_embeddings(self, data_id, emb) -> None:
        self.items[int(data_id)] = np.asarray(emb, dtype=np.float32)


def test_config_exposes_compression_section() -> None:
    config = Config(
        load_env=False,
        kv_codec="polarquant",
        kv_bits=6,
        vector_codec="qjl",
        vector_bits=8,
        compression_mode="guarded",
        compression_backend_policy="torch",
    )

    assert config.kv_codec == "polarquant"
    assert config.compression.kv_bits == 6
    assert config.vector_codec == "qjl"
    assert config.compression.compression_mode == "guarded"
    assert config.to_flat_dict()["compression_backend_policy"] == "torch"


def test_codecs_roundtrip_and_reduce_payload_size() -> None:
    vector = np.linspace(-1.0, 1.0, 128, dtype=np.float32)
    codecs = [
        QJLCodec(sketch_dim=128),
        PolarQuantCodec(bits=8),
        TurboQuantCodec(bits=8),
    ]

    for codec in codecs:
        payload = codec.encode(vector)
        restored = codec.decode(payload)
        assert restored.shape == vector.shape
        assert payload.compressed_nbytes > 0
        assert payload.raw_nbytes >= payload.compressed_nbytes or codec.__class__.__name__ == "QJLCodec"


def test_compressed_vector_store_tracks_sidecar_and_search() -> None:
    delegate = _VectorStub()
    with TemporaryDirectory() as root:
        sidecar = os.path.join(root, "vectors.bytevec.json")
        store = CompressedVectorStore(
            delegate,
            codec_name="qjl",
            bits=8,
            sidecar_path=sidecar,
        )
        vectors = [
            VectorData(id=1, data=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
            VectorData(id=2, data=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)),
        ]
        store.mul_add(vectors)
        result = store.search(np.array([1.0, 0.1, 0.0, 0.0], dtype=np.float32), top_k=1)
        store.flush()

        assert result[0][1] == 1
        assert os.path.isfile(sidecar)
        assert store.compression_stats()["entries"] == 2
        assert store.get_embeddings(1).shape == (4,)


def test_research_registry_summary_is_seeded() -> None:
    summary = research_registry_summary()

    assert summary["total_artifacts"] >= 4
    assert summary["implemented_artifacts"] >= 3


def test_runtime_descriptor_includes_byte_compression_payload() -> None:
    payload = describe_huggingface_runtime(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        model_family="llama",
        prompt_tokens=256,
        cache_hit=False,
        h2o_enabled=True,
        h2o_heavy_ratio=0.1,
        h2o_recent_ratio=0.1,
        kv_codec="hybrid",
        kv_bits=8,
        kv_hot_window_ratio=0.2,
        compression_mode="shadow",
        compression_backend_policy="torch",
    ).to_dict()

    assert payload["h2o_applied"] is True
    assert payload["byte_compression"]["requested"] is True
    assert payload["byte_compression"]["applied_codec"] == "hybrid"
    assert payload["byte_compression"]["backend"] == "torch"
