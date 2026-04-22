from __future__ import annotations

import doctest

from byte.processor import post


def test_post_module_examples_are_valid_python() -> None:
    result = doctest.testmod(post, raise_on_error=False)

    assert result.failed == 0
