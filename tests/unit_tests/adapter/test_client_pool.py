from byte.adapter.client_pool import clear_client_pools, get_async_client, get_sync_client


def setup_function() -> None:
    clear_client_pools()


def test_sync_client_pool_reuses_client_for_same_provider_and_kwargs() -> object:
    calls = []

    def factory(**kwargs) -> object:
        calls.append(kwargs)
        return {"kwargs": kwargs}

    first = get_sync_client("openai", factory, api_key="key-1", base_url="https://example.com")
    second = get_sync_client("openai", factory, base_url="https://example.com", api_key="key-1")

    assert first is second
    assert len(calls) == 1


def test_async_client_pool_reuses_client_for_same_provider_and_kwargs() -> object:
    calls = []

    def factory(**kwargs) -> object:
        calls.append(kwargs)
        return {"kwargs": kwargs}

    first = get_async_client("gemini", factory, api_key="key-2")
    second = get_async_client("gemini", factory, api_key="key-2")

    assert first is second
    assert len(calls) == 1
