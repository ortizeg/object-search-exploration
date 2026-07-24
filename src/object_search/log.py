"""Loguru configuration for the whole project.

Logging policy (INFRA-05): Loguru only. ``print()`` and stdlib ``logging`` are both lint
errors -- ruff ``T20`` catches the former, a ``flake8-tidy-imports`` banned-api entry
catches the latter. Library modules do ``from loguru import logger`` and simply log; they
never configure a sink.

Sink configuration lives here and here only.
"""

import sys
from pathlib import Path

from loguru import logger

# Explicit formats so log lines are stable across sinks and greppable in CI output.
_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure Loguru sinks for this process.

    Call this EXACTLY ONCE, at an entry point -- the CLI ``main()``, the FastAPI lifespan
    handler, or a test fixture. Never call it from a library module: ``logger.add()`` inside
    an imported module appends a fresh handler for every importer, so a single log line gets
    emitted once per import and the duplication stays invisible until output is unreadable.

    The first thing this does is ``logger.remove()``, which drops every existing handler
    (including Loguru's default stderr sink). That makes the function idempotent: calling it
    twice reconfigures rather than accumulating handlers.

    Args:
        level: Minimum level for both sinks, e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``.
        log_file: Optional path for a rotating file sink. Parent directories are created if
            missing. Rotates at 100 MB, retains 7 days, compresses rotated files as gzip.
            When ``None`` (the default) only the stderr sink is installed.
    """
    logger.remove()

    logger.add(
        sys.stderr,
        level=level,
        format=_CONSOLE_FORMAT,
        colorize=True,
    )

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=level,
            format=_FILE_FORMAT,
            rotation="100 MB",
            retention="7 days",
            compression="gz",
            enqueue=True,
        )
