import time
from typing import Any

from byte import cache
from byte.telemetry import get_log_time_func, telemetry_stage_span


def time_cal(func, func_name=None, report_func=None, chat_cache=None, span_attributes=None) -> Any:
    def inner(*args, **kwargs) -> Any:
        operation_name = func.__name__ if func_name is None else func_name
        time_start = time.time()
        with telemetry_stage_span(
            operation_name,
            chat_cache=chat_cache,
            report_func=report_func,
            fallback_cache=cache,
            attributes=span_attributes,
        ):
            res = func(*args, **kwargs)
        delta_time = time.time() - time_start
        log_time_func = get_log_time_func(
            chat_cache=chat_cache,
            report_func=report_func,
            fallback_cache=cache,
        )
        if log_time_func:
            log_time_func(operation_name, delta_time)
        if report_func is not None:
            report_func(delta_time)
        return res

    return inner
