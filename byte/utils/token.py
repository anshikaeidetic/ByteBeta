from typing import Any

from byte.utils import import_tiktoken

_encoding = None


class _FallbackEncoding:
    def encode(self, text) -> Any:
        value = str(text or "")
        if not value:
            return []
        approx_tokens = max(1, int(round(len(value) / 4.0)))
        return [0] * approx_tokens


def _get_encoding() -> Any:
    global _encoding
    if _encoding is None:
        try:
            import_tiktoken()
            import tiktoken  # pylint: disable=C0415

            _encoding = tiktoken.get_encoding("cl100k_base")
        except ModuleNotFoundError:
            _encoding = _FallbackEncoding()
    return _encoding


def token_counter(text) -> Any:
    """Token Counter"""
    num_tokens = len(_get_encoding().encode(text))
    return num_tokens
