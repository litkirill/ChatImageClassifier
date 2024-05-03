"""Logger setup for the application.

This module configures and provides a logger for the application using the loguru library.
It sets up logging to file with specific formatting, rotation, and retention policies.
"""

from loguru import logger


def setup_logger():
    """Set up and configure the logger."""
    logger.add(
        "logs/app_log.log",
        format="{time} {level} {message}",
        level="ERROR",
        rotation="10 MB",
        retention="10 days"
    )
    return logger


logger = setup_logger()
