from byte.embedding.string import to_embeddings


def test_embedding() -> None:
    message = to_embeddings("foo")
    assert message == "foo"
