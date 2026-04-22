from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def pack_unsigned(values: Iterable[int], bits: int) -> bytes:
    bit_width = int(bits or 0)
    if bit_width < 1 or bit_width > 16:
        raise ValueError("bits must be between 1 and 16")
    mask = (1 << bit_width) - 1
    output = bytearray()
    buffer = 0
    buffered_bits = 0
    for value in values:
        buffer |= (int(value) & mask) << buffered_bits
        buffered_bits += bit_width
        while buffered_bits >= 8:
            output.append(buffer & 0xFF)
            buffer >>= 8
            buffered_bits -= 8
    if buffered_bits:
        output.append(buffer & 0xFF)
    return bytes(output)


def unpack_unsigned(payload: bytes, *, count: int, bits: int) -> np.ndarray:
    total = max(int(count or 0), 0)
    bit_width = int(bits or 0)
    if total == 0:
        return np.zeros((0,), dtype=np.int32)
    if bit_width < 1 or bit_width > 16:
        raise ValueError("bits must be between 1 and 16")
    data = memoryview(payload or b"")
    values = np.zeros(total, dtype=np.int32)
    buffer = 0
    buffered_bits = 0
    cursor = 0
    mask = (1 << bit_width) - 1
    for index in range(total):
        while buffered_bits < bit_width:
            if cursor >= len(data):
                raise ValueError("packed payload is shorter than expected")
            buffer |= int(data[cursor]) << buffered_bits
            buffered_bits += 8
            cursor += 1
        values[index] = buffer & mask
        buffer >>= bit_width
        buffered_bits -= bit_width
    return values


def packed_nbytes(count: int, bits: int) -> int:
    total = max(int(count or 0), 0)
    bit_width = max(int(bits or 0), 0)
    if total <= 0 or bit_width <= 0:
        return 0
    return int(math.ceil((total * bit_width) / 8.0))
