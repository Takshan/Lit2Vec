import logging
import os
from datetime import datetime

# ANSI escape sequences for colored output
COLORS = {
    logging.DEBUG: "\x1b[36;20m",    # Cyan
    logging.INFO: "\x1b[37;20m",     # White
    logging.WARNING: "\x1b[33;20m",  # Yellow
    logging.ERROR: "\x1b[31;20m",    # Red
    logging.CRITICAL: "\x1b[31;1m",  # Bold Red
}
RESET = "\x1b[0m"


class ColorFormatter(logging.Formatter):
    """Custom formatter to add colors and file information to log messages."""
    def __init__(self):
        fmt = (
            "%(asctime)s | %(levelname)-8s | "
            "%(filename)s:%(lineno)d | "
            "%(message)s"
        )
        super().__init__(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
        self.FORMATS = {
            level: f"{color}{self._fmt}{RESET}"
            for level, color in COLORS.items()
        }

    def format(self, record):
        """Format log messages with color and file information."""
        log_fmt = self.FORMATS.get(record.levelno, self._fmt)
        formatter = logging.Formatter(log_fmt, datefmt=self.datefmt)
        return formatter.format(record)


def setup_logger(
    log_level: int | str = logging.INFO,
    log_dir: str = "logs",
    console: bool = False,
) -> logging.Logger:
    """Set up the lit2vec logger.

    By default logs are written only to a file so the Rich CLI can own the
    console. Pass ``console=True`` to also emit colored logs to stdout.
    """
    logger = logging.getLogger("lit2vec")
    logger.setLevel(log_level)

    # Avoid adding handlers more than once
    if logger.handlers:
        return logger

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColorFormatter())
        logger.addHandler(console_handler)

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f"lit2vec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)

    logging.captureWarnings(True)

    return logger
