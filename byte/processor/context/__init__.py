from typing import Any

from byte.utils.lazy_import import LazyImport

summarization = LazyImport(
    "summarization_context",
    globals(),
    "byte.processor.context.summarization_context",
)
selective = LazyImport("selective_context", globals(), "byte.processor.context.selective_context")
concat = LazyImport("concat_context", globals(), "byte.processor.context.concat_context")


__all__ = [
    "ConcatContextProcess",
    "SelectiveContextProcess",
    "SummarizationContextProcess",
]


def SummarizationContextProcess(model_name=None, tokenizer=None, target_length=512) -> Any:
    return summarization.SummarizationContextProcess(model_name, tokenizer, target_length)


def SelectiveContextProcess(
    model_type: str = "gpt2",
    lang: str = "en",
    reduce_ratio: float = 0.35,
    reduce_level: str = "phrase",
) -> Any:
    return selective.SelectiveContextProcess(
        model_type=model_type,
        lang=lang,
        reduce_ratio=reduce_ratio,
        reduce_level=reduce_level,
    )


def ConcatContextProcess() -> Any:
    return concat.ConcatContextProcess()
