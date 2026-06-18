"""
Shared logging setup. Import get_logger in any module instead of using print().
"""
import logging
import os
import sys

# I-12: honour LOG_LEVEL env var; fall back to INFO when unset or invalid
_env_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
_DEFAULT_LEVEL = getattr(logging, _env_level_name, logging.INFO)


def get_logger(name: str, level: int = _DEFAULT_LEVEL) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
