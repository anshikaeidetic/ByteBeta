from collections.abc import Callable, Sequence
from typing import Any


def batch_embed(
    embedding_func: Callable,
    inputs: Sequence[Any],
    *,
    extra_param: Any | None = None,
) -> list[Any]:
    items = list(inputs or [])
    if not items:
        return []

    index_by_key: dict[str, list[int]] = {}
    deduped_values: list[Any] = []
    for index, value in enumerate(items):
        key = _stable_key(value)
        if key not in index_by_key:
            index_by_key[key] = []
            deduped_values.append(value)
        index_by_key[key].append(index)

    deduped_embeddings = _call_embedding_func(
        embedding_func, deduped_values, extra_param=extra_param
    )
    resolved = [None for _ in items]
    for value, embedding in zip(deduped_values, deduped_embeddings):
        for index in index_by_key[_stable_key(value)]:
            resolved[index] = embedding
    return resolved


def batch_rerank(
    rerank_func: Callable,
    *,
    query: Any,
    candidates: Sequence[Any],
    batch_size: int = 16,
    extra_param: Any | None = None,
) -> list[Any]:
    items = list(candidates or [])
    if not items:
        return []
    outputs: list[Any] = []
    for start in range(0, len(items), max(1, int(batch_size or 1))):
        batch = items[start : start + max(1, int(batch_size or 1))]
        try:
            result = rerank_func(query, batch, extra_param=extra_param)
        except TypeError:
            result = rerank_func(query, batch)
        if isinstance(result, list):
            outputs.extend(result)
        else:
            outputs.append(result)
    return outputs


def _call_embedding_func(
    embedding_func: Callable, values: list[Any], *, extra_param: Any | None
) -> list[Any]:
    try:
        if extra_param is not None:
            result = embedding_func(values, extra_param=extra_param)
        else:
            result = embedding_func(values)
        if isinstance(result, list) and len(result) == len(values):
            return result
        if (
            hasattr(result, "shape")
            and getattr(result, "shape", None)
            and len(result) == len(values)
        ):
            return list(result)
    except TypeError:
        pass
    except Exception:
        pass

    outputs = []
    for value in values:
        if extra_param is not None:
            outputs.append(embedding_func(value, extra_param=extra_param))
        else:
            outputs.append(embedding_func(value))
    return outputs


def _stable_key(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    return repr(value)
