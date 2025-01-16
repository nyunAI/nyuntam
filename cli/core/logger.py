import logging
from pathlib import Path


def init_logger(log_file_path: Path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=log_file_path,
    )
