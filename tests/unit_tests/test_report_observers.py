from byte.report import Report


class _Observer:
    def __init__(self) -> None:
        self.operations = []
        self.cache_hits = 0

    def record_operation(self, operation, delta_time) -> None:
        self.operations.append((operation, delta_time))

    def record_cache_hit(self) -> None:
        self.cache_hits += 1


def test_report_notifies_observers_for_operations_and_hits() -> None:
    report = Report()
    observer = _Observer()
    report.add_observer(observer)

    report.embedding(0.25)
    report.llm(0.5)
    report.hint_cache()

    assert observer.operations == [("embedding", 0.25), ("llm", 0.5)]
    assert observer.cache_hits == 1
