"""KV-cache compression helpers for the H2O runtime."""

from __future__ import annotations

# ruff: noqa: F401
from dataclasses import dataclass
from typing import Any

from byte.h2o._runtime_common import (
    CompressionDescriptor,
    RuntimeDescriptor,
    _codec_for_tensor,
    _distortion_threshold,
    _normalize_kv_codec,
    _past_seq_len,
    _tensor_to_numpy,
)


class _CompressedTensorState:
    original_shape: tuple[int, ...]
    leading_shape: tuple[int, ...]
    cold_len: int
    feature_dim: int
    payloads: list[Any]
    hot_tensor: Any | None
    tail: tuple[Any, ...] = ()
    raw_bytes: int = 0
    compressed_bytes: int = 0
    distortion: float = 0.0
    dtype_name: str = "float32"


class _KVCompressionController:
    def __init__(self, descriptor: RuntimeDescriptor, *, device: Any, torch_module: Any) -> None:
        self.descriptor = descriptor
        self.device = device
        self.torch = torch_module
        self._raw_past_key_values = None
        self._compressed_layers: list[tuple[_CompressedTensorState, _CompressedTensorState, tuple[Any, ...]]] = []
        self._seq_len = 0

    def capture(self, past_key_values: Any) -> None:
        self._seq_len = _past_seq_len(past_key_values)
        compression = self.descriptor.compression
        if compression is None or not compression.applied or not past_key_values:
            self._raw_past_key_values = past_key_values
            return
        try:
            compressed_layers, metrics = self._compress_past_key_values(past_key_values)
            compression.layer_count = int(metrics.get("layer_count", 0) or 0)
            compression.compressed_layers = int(metrics.get("compressed_layers", 0) or 0)
            compression.raw_bytes = int(metrics.get("raw_bytes", 0) or 0)
            compression.compressed_bytes = int(metrics.get("compressed_bytes", 0) or 0)
            compression.compression_ratio = float(metrics.get("compression_ratio", 0.0) or 0.0)
            compression.distortion_mean = float(metrics.get("distortion_mean", 0.0) or 0.0)
            if compression.mode == "shadow":
                self._raw_past_key_values = past_key_values
                self._compressed_layers = compressed_layers
                return
            if (
                compression.mode == "guarded"
                and compression.distortion_mean > _distortion_threshold(compression.applied_codec)
            ):
                compression.applied = False
                compression.fallback_reason = (
                    f"distortion_guard:{compression.distortion_mean:.4f}"
                )
                self._raw_past_key_values = past_key_values
                self._compressed_layers = []
                return
            self._raw_past_key_values = None
            self._compressed_layers = compressed_layers
        except Exception as exc:  # pylint: disable=W0703
            compression.applied = False
            compression.fallback_reason = f"compression_capture_failed:{type(exc).__name__}"
            self._raw_past_key_values = past_key_values
            self._compressed_layers = []

    def materialize(self) -> Any:
        compression = self.descriptor.compression
        if self._raw_past_key_values is not None:
            return self._raw_past_key_values
        if compression is None or not compression.applied or not self._compressed_layers:
            return self._raw_past_key_values
        try:
            return self._decode_past_key_values(self._compressed_layers)
        except Exception as exc:  # pylint: disable=W0703
            compression.applied = False
            compression.fallback_reason = f"compression_decode_failed:{type(exc).__name__}"
            return self._raw_past_key_values

    def seq_len(self) -> int:
        return self._seq_len

    def _compress_past_key_values(self, past_key_values: Any) -> tuple[list[Any], dict[str, Any]]:
        compression = self.descriptor.compression
        assert compression is not None
        hot_ratio = max(0.0, min(float(compression.hot_window_ratio or 0.0), 1.0))
        compressed_layers = []
        raw_bytes = 0
        compressed_bytes = 0
        distortions = []
        compressed_layer_count = 0
        for layer in past_key_values:
            key_state = layer[0]
            value_state = layer[1]
            key_payload = self._compress_tensor(
                key_state,
                codec_name=compression.applied_codec,
                bits=compression.bits,
                hot_ratio=hot_ratio,
                is_key=True,
            )
            value_payload = self._compress_tensor(
                value_state,
                codec_name=compression.applied_codec,
                bits=compression.bits,
                hot_ratio=hot_ratio,
                is_key=False,
            )
            raw_bytes += key_payload.raw_bytes + value_payload.raw_bytes
            compressed_bytes += key_payload.compressed_bytes + value_payload.compressed_bytes
            distortions.extend([key_payload.distortion, value_payload.distortion])
            if key_payload.compressed_bytes or value_payload.compressed_bytes:
                compressed_layer_count += 1
            compressed_layers.append((key_payload, value_payload, tuple(layer[2:])))
        metrics = {
            "layer_count": len(compressed_layers),
            "compressed_layers": compressed_layer_count,
            "raw_bytes": raw_bytes,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": round(float(compressed_bytes) / float(raw_bytes), 6)
            if raw_bytes > 0
            else 0.0,
            "distortion_mean": round(
                sum(distortions) / len([item for item in distortions if item >= 0]),
                6,
            )
            if distortions
            else 0.0,
        }
        return compressed_layers, metrics

    def _compress_tensor(
        self,
        tensor: Any,
        *,
        codec_name: str,
        bits: int,
        hot_ratio: float,
        is_key: bool,
    ) -> _CompressedTensorState:
        array = _tensor_to_numpy(tensor)
        original_shape = tuple(int(item) for item in getattr(array, "shape", ()))
        if len(original_shape) < 2:
            return _CompressedTensorState(
                original_shape=original_shape,
                leading_shape=tuple(),
                cold_len=0,
                feature_dim=0,
                payloads=[],
                hot_tensor=tensor,
                raw_bytes=int(getattr(array, "nbytes", 0) or 0),
                compressed_bytes=int(getattr(array, "nbytes", 0) or 0),
                dtype_name=str(getattr(tensor, "dtype", "float32")).replace("torch.", ""),
            )
        seq_len = int(original_shape[-2])
        feature_dim = int(original_shape[-1])
        hot_window = min(seq_len, max(0, int(round(seq_len * hot_ratio))))
        cold_len = max(seq_len - hot_window, 0)
        leading_shape = tuple(original_shape[:-2])
        leading = 1
        for item in leading_shape:
            leading *= int(item)
        reshaped = array.reshape((leading, seq_len, feature_dim))
        codec = _codec_for_tensor(codec_name, bits, is_key=is_key)
        payloads = []
        compressed_bytes = 0
        distortions = []
        for index in range(leading):
            block = reshaped[index, :cold_len, :]
            if cold_len <= 0:
                payloads.append(None)
                continue
            payload = codec.encode(block)
            payloads.append(payload)
            compressed_bytes += int(getattr(payload, "compressed_nbytes", 0) or 0)
            reconstructed = codec.decode(payload).reshape(block.shape)
            baseline = float(((block.astype("float32") ** 2).sum()) ** 0.5)
            error = float((((reconstructed - block) ** 2).sum()) ** 0.5)
            distortions.append(error / baseline if baseline > 0 else 0.0)
        hot_tensor = None
        if hot_window > 0:
            hot_tensor = tensor[..., -hot_window:, :].detach().clone()
            compressed_bytes += int(getattr(_tensor_to_numpy(hot_tensor), "nbytes", 0) or 0)
        return _CompressedTensorState(
            original_shape=original_shape,
            leading_shape=leading_shape,
            cold_len=cold_len,
            feature_dim=feature_dim,
            payloads=payloads,
            hot_tensor=hot_tensor,
            raw_bytes=int(getattr(array, "nbytes", 0) or 0),
            compressed_bytes=compressed_bytes,
            distortion=round(sum(distortions) / len(distortions), 6) if distortions else 0.0,
            dtype_name=str(getattr(tensor, "dtype", "float32")).replace("torch.", ""),
        )

    def _decode_past_key_values(self, compressed_layers: list[Any]) -> Any:
        restored = []
        compression = self.descriptor.compression
        assert compression is not None
        for key_payload, value_payload, tail in compressed_layers:
            key_state = self._decode_tensor(
                key_payload,
                codec_name=compression.applied_codec,
                bits=compression.bits,
                is_key=True,
            )
            value_state = self._decode_tensor(
                value_payload,
                codec_name=compression.applied_codec,
                bits=compression.bits,
                is_key=False,
            )
            restored.append((key_state, value_state, *tail))
        return tuple(restored)

    def _decode_tensor(
        self,
        state: _CompressedTensorState,
        *,
        codec_name: str,
        bits: int,
        is_key: bool,
    ) -> Any:
        if state.cold_len <= 0:
            if state.hot_tensor is None:
                return None
            return state.hot_tensor.to(
                device=self.device,
                dtype=getattr(self.torch, state.dtype_name, self.torch.float32),
            )
        codec = _codec_for_tensor(codec_name, bits, is_key=is_key)
        target_dtype = getattr(self.torch, state.dtype_name, self.torch.float32)
        leading = 1
        for item in state.leading_shape:
            leading *= int(item)
        cold_blocks = []
        for payload in state.payloads:
            if payload is None:
                cold_blocks.append(
                    self.torch.zeros(
                        (0, state.feature_dim),
                        device=self.device,
                        dtype=target_dtype,
                    )
                )
                continue
            decoded = codec.decode(payload)
            cold_blocks.append(
                self.torch.as_tensor(decoded, device=self.device, dtype=target_dtype)
            )
        cold_tensor = self.torch.stack(cold_blocks, dim=0).reshape(
            state.leading_shape + (state.cold_len, state.feature_dim)
        )
        if state.hot_tensor is None:
            return cold_tensor
        hot_tensor = state.hot_tensor.to(device=self.device, dtype=target_dtype)
        return self.torch.cat((cold_tensor, hot_tensor), dim=-2)


__all__ = ["_CompressedTensorState", "_KVCompressionController"]
