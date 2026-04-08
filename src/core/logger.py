"""Logging utilities for Cloud-ICU Sentinel."""

from __future__ import annotations

import logging
from logging import Logger

from src.core.config import Settings


def configure_logging(settings: Settings) -> Logger:
    """Configure and return the application root logger.

    Args:
        settings: Loaded application settings.

    Returns:
        Logger: Configured application logger.
    """

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger: Logger = logging.getLogger("cloud_icu_sentinel")
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    return logger
