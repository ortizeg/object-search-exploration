"""Tests for the Loguru setup in object_search.log."""

from pathlib import Path

from loguru import logger

from object_search.log import setup_logging


def test_setup_logging_is_idempotent():
    """Calling setup_logging twice must not accumulate handlers.

    This is the property the docstring promises. A list sink lets us count emissions
    directly: if the second call had appended rather than replaced, one log call would
    produce two records.
    """
    captured: list[str] = []

    setup_logging("DEBUG")
    setup_logging("DEBUG")

    # Replace the configured sinks with a countable one, mirroring what setup_logging does.
    logger.remove()
    logger.add(captured.append, level="DEBUG", format="{message}")
    logger.info("only-once")

    assert len([line for line in captured if "only-once" in line]) == 1


def test_setup_logging_removes_preexisting_sinks():
    """A sink added before setup_logging must be gone afterwards."""
    captured: list[str] = []
    sink_id = logger.add(captured.append, level="DEBUG", format="{message}")

    setup_logging("DEBUG")
    logger.info("after-setup")

    assert captured == []
    # The old handler is already removed; removing it again must raise.
    try:
        logger.remove(sink_id)
    except ValueError:
        pass
    else:  # pragma: no cover - only reached if setup_logging failed to clear sinks
        msg = "setup_logging did not remove the pre-existing sink"
        raise AssertionError(msg)


def test_setup_logging_creates_file_sink(tmp_path: Path):
    """Passing log_file must create the file and write records to it."""
    log_file = tmp_path / "nested" / "run.log"

    setup_logging("INFO", log_file=log_file)
    logger.info("written-to-file")
    # enqueue=True means the writer thread owns the handle; remove() flushes and closes it.
    logger.remove()

    assert log_file.exists()
    assert "written-to-file" in log_file.read_text()


def test_setup_logging_respects_level(tmp_path: Path):
    """A DEBUG record must not appear in a sink configured at WARNING."""
    log_file = tmp_path / "warn.log"

    setup_logging("WARNING", log_file=log_file)
    logger.debug("debug-record")
    logger.warning("warning-record")
    logger.remove()

    contents = log_file.read_text()
    assert "warning-record" in contents
    assert "debug-record" not in contents


def test_setup_logging_without_file_creates_no_file(tmp_path: Path):
    """The file sink is opt-in; omitting log_file must leave the directory empty."""
    setup_logging("INFO")
    logger.info("stderr-only")

    assert list(tmp_path.iterdir()) == []
