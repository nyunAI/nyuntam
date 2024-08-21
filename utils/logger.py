import logging
from typing import Union, Optional
from pathlib import Path
import sys

LOG_FILE_NAME = "log.log"


class LogFile(object):
    """File-like object to log text using the `logging` module."""

    def __init__(self, name=None):
        self.logger = logging.getLogger(name)

    def write(self, msg, level=logging.INFO):
        if msg != "\n":
            self.logger.log(level, msg)

    def flush(self):
        for handler in self.logger.handlers:
            handler.flush()


def set_logger(
    logging_path: Optional[Union[str, Path]] = None,
    stream_stdout: Optional[bool] = None,
) -> None:
    """Set the logger for the module. Clears all handlers defined by previous loggers.
    The logger will be saved to the specified logging directory (or file).

    Args:
        logging_path (Optional[Union[str, Path]]): The path to the logging directory (or file).
    """
    # clears handlers defined by previous logger(s)
    logging.root.handlers.clear()

    if logging_path:
        if not isinstance(logging_path, Path):
            logging_path = Path(logging_path)

        if logging_path.is_dir():
            parent = logging_path
            logging_path = logging_path / LOG_FILE_NAME
        else:
            parent = logging_path.parent

        parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename=logging_path,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H-%M-%S",
            level=logging.INFO,
            force=True,  # force the configuration over any existing configuration
        )

    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H-%M-%S",
            level=logging.INFO,
            force=True,  # force the configuration over any existing configuration
        )

    if stream_stdout:
        sys.stdout = LogFile("stdout")
