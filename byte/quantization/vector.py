from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from byte.manager.data_manager import SSDataManager
from byte.manager.vector_data.base import VectorBase, VectorData
from byte.quantization.features import blend_token_streams, hashed_text_features
from byte.quantization.polar import PolarQuantCodec, PolarQuantPayload
from byte.quantization.qjl import QJLCodec, QJLPayload
from byte.quantization.turbo import TurboQuantCodec, TurboQuantPayload


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def build_vector_codec(
    codec_name: str,
    *,
    bits: int,
    dimension_hint: int = 0,
) -> Any | None:
    normalized = str(codec_name or "").strip().lower()
    if normalized in {"", "disabled"}:
        return None
    if normalized == "qjl":
        sketch_dim = min(max(int(dimension_hint or 0), 32), 256) if dimension_hint else 128
        return QJLCodec(sketch_dim=sketch_dim)
    if normalized == "turboquant":
        return TurboQuantCodec(bits=bits)
    if normalized == "polarquant":
        return PolarQuantCodec(bits=bits)
    return None


@dataclass(frozen=True)
class VectorCodecStats:
    codec: str
    bits: int
    raw_bytes: int
    compressed_bytes: int
    entries: int
    lookups: int
    sidecar_hits: int

    def to_dict(self) -> dict[str, Any]:
        ratio = (
            round(float(self.compressed_bytes) / float(self.raw_bytes), 6)
            if self.raw_bytes > 0
            else 0.0
        )
        return {
            "codec": self.codec,
            "bits": self.bits,
            "raw_bytes": self.raw_bytes,
            "compressed_bytes": self.compressed_bytes,
            "compression_ratio": ratio,
            "entries": self.entries,
            "lookups": self.lookups,
            "sidecar_hits": self.sidecar_hits,
        }


class CompressedVectorStore(VectorBase):
    def __init__(
        self,
        delegate: VectorBase,
        *,
        codec_name: str,
        bits: int,
        sidecar_path: str = "",
    ) -> None:
        self.delegate = delegate
        self.codec_name = str(codec_name or "").strip().lower()
        self.bits = max(int(bits or 0), 1)
        self.sidecar_path = str(sidecar_path or "").strip()
        self.codec = build_vector_codec(self.codec_name, bits=self.bits)
        self._payloads: dict[int, Any] = {}
        self._raw_bytes = 0
        self._compressed_bytes = 0
        self._lookups = 0
        self._sidecar_hits = 0
        if self.sidecar_path:
            self._load_sidecar()

    def mul_add(self, datas: list[VectorData]) -> Any | None:
        if not datas:
            return
        if self.codec is None:
            return self.delegate.mul_add(datas)
        for item in datas:
            vector = _to_numpy(item.data).astype(np.float32).reshape(-1)
            payload = self.codec.encode(vector)
            self._payloads[int(item.id)] = payload
            self._raw_bytes += int(vector.nbytes)
            self._compressed_bytes += int(getattr(payload, "compressed_nbytes", vector.nbytes))
        return self.delegate.mul_add(datas)

    def search(self, data: np.ndarray, top_k: int) -> Any:
        self._lookups += 1
        if self.codec is None or not self._payloads:
            return self.delegate.search(data=data, top_k=top_k)
        requested_top_k = max(int(top_k if top_k not in (None, -1) else 1), 1)
        query = _to_numpy(data).reshape(-1).astype(np.float32)
        ranked = []
        for vector_id, payload in self._payloads.items():
            similarity = float(self.codec.similarity(query, payload))
            ranked.append((1.0 - similarity, int(vector_id)))
        ranked.sort(key=lambda item: item[0])
        if ranked:
            self._sidecar_hits += 1
            return ranked[:requested_top_k]
        return self.delegate.search(data=data, top_k=top_k)

    def rebuild(self, ids=None) -> bool:
        return self.delegate.rebuild(ids=ids)

    def delete(self, ids) -> bool:
        for item in ids or []:
            self._payloads.pop(int(item), None)
        return self.delegate.delete(ids)

    def flush(self) -> None:
        self.delegate.flush()
        self._flush_sidecar()

    def close(self) -> None:
        self.flush()
        self.delegate.close()

    def get_embeddings(self, data_id) -> Any:
        payload = self._payloads.get(int(data_id))
        if payload is not None and self.codec is not None:
            return self.codec.decode(payload).reshape(-1).astype(np.float32)
        return self.delegate.get_embeddings(data_id)

    def update_embeddings(self, data_id, emb: np.ndarray) -> Any:
        vector = _to_numpy(emb).reshape(-1).astype(np.float32)
        if self.codec is not None:
            prior = self._payloads.get(int(data_id))
            if prior is not None:
                self._compressed_bytes -= int(getattr(prior, "compressed_nbytes", 0))
            payload = self.codec.encode(vector)
            self._payloads[int(data_id)] = payload
            self._compressed_bytes += int(getattr(payload, "compressed_nbytes", vector.nbytes))
        return self.delegate.update_embeddings(data_id, emb)

    def compression_stats(self) -> dict[str, Any]:
        return VectorCodecStats(
            codec=self.codec_name,
            bits=self.bits,
            raw_bytes=self._raw_bytes,
            compressed_bytes=self._compressed_bytes,
            entries=len(self._payloads),
            lookups=self._lookups,
            sidecar_hits=self._sidecar_hits,
        ).to_dict()

    def _flush_sidecar(self) -> None:
        if not self.sidecar_path:
            return
        payload = {
            "codec": self.codec_name,
            "bits": self.bits,
            "stats": self.compression_stats(),
            "items": {
                str(vector_id): _payload_to_json(vector_payload)
                for vector_id, vector_payload in self._payloads.items()
            },
        }
        os.makedirs(os.path.dirname(self.sidecar_path) or ".", exist_ok=True)
        with open(self.sidecar_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True)

    def _load_sidecar(self) -> None:
        if not self.sidecar_path or not os.path.isfile(self.sidecar_path):
            return
        try:
            with open(self.sidecar_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            for vector_id, item in (payload.get("items", {}) or {}).items():
                decoded = _payload_from_json(item, codec=self.codec)
                if decoded is not None:
                    self._payloads[int(vector_id)] = decoded
        except Exception:
            self._payloads = {}


def maybe_wrap_vector_data_manager(data_manager: Any, config: Any) -> Any:
    if not isinstance(data_manager, SSDataManager):
        return data_manager
    codec_name = str(getattr(config, "vector_codec", "disabled") or "disabled").strip().lower()
    if codec_name in {"", "disabled"}:
        return data_manager
    if isinstance(getattr(data_manager, "v", None), CompressedVectorStore):
        return data_manager
    sidecar_path = ""
    delegate = data_manager.v
    index_path = getattr(delegate, "_index_file_path", "")
    if index_path:
        sidecar_path = f"{index_path}.bytevec.json"
    data_manager.v = CompressedVectorStore(
        delegate,
        codec_name=codec_name,
        bits=int(getattr(config, "vector_bits", 8) or 8),
        sidecar_path=sidecar_path,
    )
    return data_manager


def compression_text_entry(
    value: Any,
    *,
    codec_name: str,
    bits: int,
    feature_dimension: int = 256,
) -> tuple[dict[str, Any], float]:
    codec = build_vector_codec(codec_name, bits=bits, dimension_hint=feature_dimension)
    features = blend_token_streams([value])
    if codec is None:
        return {}, 0.0
    payload = codec.encode(features)
    raw_bytes = int(features.nbytes)
    compressed_bytes = int(getattr(payload, "compressed_nbytes", raw_bytes))
    ratio = round(float(compressed_bytes) / float(raw_bytes), 6) if raw_bytes > 0 else 0.0
    summary = payload.summary()
    summary["compression_ratio"] = ratio
    summary["feature_dimension"] = feature_dimension
    return summary, ratio


def related_text_score(
    query_text: str,
    payload: Any,
    *,
    codec_name: str,
    bits: int,
    feature_dimension: int = 256,
) -> float:
    codec = build_vector_codec(codec_name, bits=bits, dimension_hint=feature_dimension)
    if codec is None or payload is None:
        return 0.0
    query = hashed_text_features(query_text, dimension=feature_dimension)
    return float(codec.similarity(query, payload))


def encode_text_payload(
    value: Any,
    *,
    codec_name: str,
    bits: int,
    feature_dimension: int = 256,
) -> Any:
    codec = build_vector_codec(codec_name, bits=bits, dimension_hint=feature_dimension)
    if codec is None:
        return None
    features = blend_token_streams([value])
    return codec.encode(features)


def _payload_to_json(payload: Any) -> dict[str, Any]:
    raw = dict(payload.__dict__)
    for key, value in list(raw.items()):
        if isinstance(value, bytes):
            raw[key] = {"__bytes__": value.hex()}
        elif hasattr(value, "__dict__") and not isinstance(value, (str, int, float, bool, list, dict, tuple)):
            raw[key] = _payload_to_json(value)
    return raw


def _payload_from_json(payload: dict[str, Any], *, codec: Any) -> Any:
    if codec is None:
        return None
    restored = {}
    for key, value in (payload or {}).items():
        if isinstance(value, dict) and "__bytes__" in value:
            restored[key] = bytes.fromhex(str(value["__bytes__"]))
        elif isinstance(value, dict) and value.get("codec"):
            restored[key] = _payload_from_json(value, codec=codec)
        elif key == "shape" and isinstance(value, list):
            restored[key] = tuple(value)
        else:
            restored[key] = value
    payload_codec = str(restored.get("codec", "") or "").strip().lower()
    if payload_codec == "qjl":
        return QJLPayload(**restored)
    if payload_codec == "polarquant":
        return PolarQuantPayload(**restored)
    if payload_codec == "turboquant":
        residual = restored.get("residual")
        if isinstance(residual, dict):
            restored["residual"] = _payload_from_json(residual, codec=codec)
        return TurboQuantPayload(**restored)
    return None
