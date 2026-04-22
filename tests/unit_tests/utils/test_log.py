from byte.utils.log import byte_log


def test_error_type() -> None:
    byte_log.setLevel("INFO")
    byte_log.error("Cache log error.")
    byte_log.warning("Cache log warning.")
    byte_log.info("Cache log info.")
    assert byte_log.level == 20
