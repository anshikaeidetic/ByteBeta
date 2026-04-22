import weakref
from typing import Any


class Report:

    def __init__(self, owner_cache=None) -> None:
        self.op_pre = OpCounter()
        self.op_embedding = OpCounter()
        self.op_search = OpCounter()
        self.op_data = OpCounter()
        self.op_evaluation = OpCounter()
        self.op_post = OpCounter()
        self.op_llm = OpCounter()
        self.op_save = OpCounter()
        self.hint_cache_count = 0
        self._observers = []
        self._owner_cache_ref = None
        self.attach_owner(owner_cache)

    def attach_owner(self, owner_cache) -> None:
        if owner_cache is None:
            self._owner_cache_ref = None
            return
        self._owner_cache_ref = weakref.ref(owner_cache)

    @property
    def owner_cache(self) -> Any | None:
        if self._owner_cache_ref is None:
            return None
        return self._owner_cache_ref()

    def add_observer(self, observer) -> None:
        """Attach a best-effort observer for telemetry sinks."""
        if observer is None or observer in self._observers:
            return
        self._observers.append(observer)

    def remove_observer(self, observer) -> None:
        if observer in self._observers:
            self._observers.remove(observer)

    def _notify_operation(self, operation: str, delta_time: float) -> None:
        for observer in list(self._observers):
            try:
                observer.record_operation(operation, delta_time)
            except Exception:  # pylint: disable=W0703
                continue

    def _notify_cache_hit(self) -> None:
        for observer in list(self._observers):
            try:
                observer.record_cache_hit()
            except Exception:  # pylint: disable=W0703
                continue

    def _record(self, operation: str, delta_time: float) -> None:
        counter: OpCounter = getattr(self, f"op_{operation}")
        counter.total_time += delta_time
        counter.count += 1
        self._notify_operation(operation, delta_time)

    def pre(self, delta_time: float) -> None:
        self._record("pre", delta_time)

    def embedding(self, delta_time: float) -> None:
        self._record("embedding", delta_time)

    def search(self, delta_time: float) -> None:
        self._record("search", delta_time)

    def data(self, delta_time: float) -> None:
        self._record("data", delta_time)

    def evaluation(self, delta_time: float) -> None:
        self._record("evaluation", delta_time)

    def post(self, delta_time: float) -> None:
        self._record("post", delta_time)

    def llm(self, delta_time: float) -> None:
        self._record("llm", delta_time)

    def save(self, delta_time: float) -> None:
        self._record("save", delta_time)

    def average_pre_time(self) -> float:
        return self.op_pre.average()

    def average_embedding_time(self) -> float:
        return self.op_embedding.average()

    def average_search_time(self) -> float:
        return self.op_search.average()

    def average_data_time(self) -> float:
        return self.op_data.average()

    def average_evaluation_time(self) -> float:
        return self.op_evaluation.average()

    def average_post_time(self) -> float:
        return self.op_post.average()

    def average_llm_time(self) -> float:
        return self.op_llm.average()

    def average_save_time(self) -> float:
        return self.op_save.average()

    def hint_cache(self) -> None:
        self.hint_cache_count += 1
        self._notify_cache_hit()


class OpCounter:

    def __init__(self) -> None:
        self.count: int = 0
        self.total_time: float = 0.0

    def average(self) -> float:
        return round(self.total_time / self.count, 4) if self.count != 0 else 0.0
