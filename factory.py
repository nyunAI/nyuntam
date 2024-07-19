from nyuntam.constants.keys import FactoryTypes, Tasks, FactoryArgumentKeys
from nyuntam.utils.logger import set_logger
from nyuntam.algorithm import Algorithm

from typing import Dict, Optional, Union, List
from abc import abstractmethod
from pathlib import Path
import logging
import json
import yaml

__all__ = ["Factory", "FactoryTypes", "register_factory"]
logger = logging.getLogger(__name__)


def get_factories(task: Union[str, FactoryTypes, Tasks]) -> List["Factory"]:
    """Get the factory classes for the given Job task.

    Args:
        task: Job task.

    Returns:
        List[Factory]: List of factory classes.
    """

    task: FactoryTypes = FactoryTypes(task)
    if task == FactoryTypes.TEXT_GENERATION:
        # text-generation
        from text_generation.main import Factory as TextGenerationFactory

        cls = [TextGenerationFactory]
    elif task == FactoryTypes.VISION:
        # vision
        from vision.factory import CompressionFactory as VisionFactory

        cls = [VisionFactory]
    else:
        raise ValueError(f"Unsupported task: {task}")

    return cls


class Factory:
    """Factory class for creating different algorithms.

    Args:
        args: Arguments for the algorithm.

    Returns:
        Factory: Factory instance.
    """

    _cls: Optional["Factory"] = None
    _type: Optional[FactoryTypes] = None

    _instance: Optional[Algorithm] = None
    """Compression algorithm instance."""

    def __init__(self, args: Dict) -> "Factory":
        """Initialize the Factory instance."""
        self.__pre_init__(args)

    def __pre_init__(self, *args, **kwargs):
        """Pre-initialization method. Use/extend this method to perform any checks before initializing the class and
        fail fast to adapt to other factory classes."""
        kw = args[0]
        assert (
            FactoryTypes(kw.get(FactoryArgumentKeys.TASK, None)) == self._type
        ), "Invalid task type."

        algorithm_name = kw.get(FactoryArgumentKeys.ALGORITHM, None)
        assert (
            self.get_algorithm(algorithm_name) is not None
        ), f"Invalid algorithm: {algorithm_name}"

    @abstractmethod
    def get_algorithm(self, name: str) -> Algorithm:
        """Get the algorithm class."""

    @classmethod
    def create_from_dict(cls, args: Dict) -> Optional[Union["Factory", None]]:
        """Create a Factory instance from a dictionary."""
        cls = get_factories(args.get(FactoryArgumentKeys.TASK))
        task = args.get(FactoryArgumentKeys.TASK)
        task: FactoryTypes = FactoryTypes(task)
        for caller in cls:
            try:
                instance = caller(args)
                return instance
            except Exception as e:
                logger.exception(f"[{cls}] {e}")
                continue
        raise Exception("Factory instance could not be created.")

    @classmethod
    def create_from_json(
        cls, path: Union[str, Path]
    ) -> Optional[Union["Factory", None]]:
        """Create a Factory instance from a JSON file."""
        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r") as f:
            args = json.load(f)
        set_logger(logging_path=args['LOGGING_PATH'])
        return cls.create_from_dict(args)

    @classmethod
    def create_from_yaml(
        cls, path: Union[str, Path]
    ) -> Optional[Union["Factory", None]]:
        """Create a Factory instance from a YAML file."""

        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r") as f:
            args = yaml.safe_load(f)
        set_logger(logging_path=args['LOGGING_PATH'])
        return cls.create_from_dict(args)

    @property
    def algorithm(self) -> Algorithm:
        return self._instance

    @algorithm.setter
    def algorithm(self, instance: Algorithm) -> None:
        self._instance = instance

    def set_logger(self, path: Union[str, Path]) -> None:
        set_logger(logging_path=path)

    def run(self) -> None:
        if self.algorithm is None:
            raise ValueError("No algorithm instance has been created.")
        self.algorithm.compress_model()