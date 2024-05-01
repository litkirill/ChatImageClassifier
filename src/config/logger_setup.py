from loguru import logger


def setup_logger():
    logger.add(
        "logs/app_log.log",
        format="{time} {level} {message}",
        level="ERROR",
        rotation="10 MB",
        retention="10 days"
    )
    return logger


logger = setup_logger()
