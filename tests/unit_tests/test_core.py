import gc
import time
import weakref

from byte import Cache, Config, cache
from byte.manager import manager_factory
from byte.report import Report
from byte.utils.cache_func import cache_all
from byte.utils.time import time_cal


def test_time_cal() -> None:
    def log_time_func(fname, delta_time) -> None:
        assert fname == "unit_test"
        assert delta_time > 0.1

    cache.config = Config(log_time_func=log_time_func)

    @time_cal
    def time_cal_annotation() -> None:
        time.sleep(0.2)

    func_name = "test_time_cal"

    def log_time_func(fname, delta_time) -> None:
        assert fname == func_name
        assert delta_time > 0.1

    cache.config = Config(log_time_func=log_time_func)

    def report_func(delta_time) -> None:
        assert delta_time > 0.1

    def time_cal_without_annotation() -> None:
        time.sleep(0.2)

    time_cal(time_cal_without_annotation, func_name=func_name, report_func=report_func)()

    cache.config = None


def test_cache_all() -> None:
    assert cache_all()


def test_report() -> None:
    report = Report()
    report.embedding(1)
    report.embedding(3)
    report.search(2)
    report.search(4)
    report.hint_cache()
    report.hint_cache()

    assert report.average_embedding_time() == 2
    assert report.op_embedding.count == 2
    assert report.average_search_time() == 3
    assert report.op_search.count == 2
    assert report.hint_cache_count == 2


def test_cache_instances_can_be_collected_after_init(tmp_path) -> None:
    cache_obj = Cache()
    cache_obj.init(data_manager=manager_factory("map", data_dir=str(tmp_path)))

    ref = weakref.ref(cache_obj)
    del cache_obj
    gc.collect()

    assert ref() is None
