import logging
from typing import Union, Optional
from pathlib import Path

LOG_FILE_NAME = "log.log"


def set_logger(logging_path: Optional[Union[str, Path]] = None) -> None:
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
