"""Structured logging configuration for Swara."""

import logging
import sys
from typing import Optional


def get_logger(name: str = "swara", level: Optional[str] = None) -> logging.Logger:
    """Returns a configured logger with clean formatted output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if level:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    elif not logger.level:
        logger.setLevel(logging.INFO)

    return logger
