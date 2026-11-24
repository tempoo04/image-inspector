"""Shared logging setup.

Call :func:`get_logger` anywhere to obtain a module logger. The first call
configures a single stream handler on the package root logger; repeat calls are
cheap and never add duplicate handlers.
"""

from __future__ import annotations

import logging

_ROOT_NAME = "vision_explorer"
_CONFIGURED = False


def _configure_root(level: int) -> None:
    global _CONFIGURED
    root = logging.getLogger(_ROOT_NAME)
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    root.setLevel(level)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a child logger under the package root, configuring output once.

    *name* is typically ``__name__``; the package prefix is stripped/normalized
    so loggers nest under ``vision_explorer``.
    """
    _configure_root(level)
    short = name.split(".")[-1] if name else "app"
    return logging.getLogger(f"{_ROOT_NAME}.{short}")
