"""Structured logging configuration.

A single :func:`configure_logging` call sets up a consistent, timestamped log
format for the whole application. Modules obtain their logger through
:func:`get_logger`, which keeps logger names aligned with the import path.
"""

from __future__ import annotations

import logging
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging once, idempotently.

    Args:
        level: Logging level (name or numeric value) for the root logger.
    """
    global _configured
    if _configured:
        logging.getLogger().setLevel(level)
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring logging on first use.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger`.
    """
    if not _configured:
        configure_logging()
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
