from byte.utils.error import (
    CacheError,
    NotFoundError,
    NotInitError,
    ParamError,
)


def test_error_type() -> None:
    not_init_error = NotInitError()
    assert issubclass(type(not_init_error), CacheError)

    not_found_store_error = NotFoundError("unittest", "test_error_type")
    assert issubclass(type(not_found_store_error), CacheError)

    param_error = ParamError("unittest")
    assert issubclass(type(param_error), CacheError)


def test_wrap() -> None:
    import openai

    from byte.utils.error import wrap_error

    def raise_error() -> None:
        try:
            raise openai.OpenAIError("test")
        except openai.OpenAIError as e:
            raise wrap_error(e) from e

    is_exception = False
    try:
        raise_error()
    except openai.OpenAIError:
        is_exception = True

    assert is_exception
