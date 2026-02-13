"""Logging configuration for virosense using loguru."""

import sys

from loguru import logger


def setup_logging(level: str = "INFO") -> None:
    """Configure loguru logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level=level,
        colorize=True,
    )


def get_logger(name: str):
    """Get a logger instance."""
    return logger.bind(module=name)
