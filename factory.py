from nyuntam.constants.keys import (
    FactoryTypes,
    Task,
    VisionTasks,
    TextGenTasks,
    FactoryArgumentKeys,
    JobServices,
)
from nyuntam.utils.logger import set_logger
from nyuntam.algorithm import Algorithm
from typing import Dict, Optional, Union, List
from abc import abstractmethod
from pathlib import Path
import logging
import json
import yaml

__all__ = ["Factory", "FactoryTypes"]
logger = logging.getLogger(__name__)


def get_factories(
    task: Union[str, Task],
    job_service: Union[str, JobServices],
) -> List["Factory"]:
    """Get the factory classes for the given Job service & task.

    Args:
        task: Job task.
        job_service: Job service.

    Returns:
        List[Factory]: List of factory classes.
    """

    if job_service == JobServices.KOMPRESS:

        if isinstance(task, str):
            task: Task = Task.create(job_service, task)

        # Kompress
        if isinstance(task, TextGenTasks):
            # text-generation
            from text_generation.main import Factory as TextGenerationFactory

            cls = [TextGenerationFactory]
        elif isinstance(task, VisionTasks):
            # vision
            from vision.factory import CompressionFactory as VisionFactory

            cls = [VisionFactory]
    elif job_service == JobServices.ADAPT:
        from nyuntam_adapt.factory import AdaptFactory

        cls = [AdaptFactory]
    else:
        raise ValueError(f"Unsupported task or job service: {task, job_service}")

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
        job_service = JobServices.get_service(
            kw.get(FactoryArgumentKeys.JOB_SERVICE, None)
        )
        task = Task.create(job_service, kw.get(FactoryArgumentKeys.TASK, None))
        factory_type = FactoryTypes.get_factory_type(job_service, task)
        assert factory_type == self._type, f"Invalid factory type: {factory_type}"

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
        job_service = JobServices.get_service(
            args.get(FactoryArgumentKeys.JOB_SERVICE, None)
        )
        task = Task.create(job_service, args.get(FactoryArgumentKeys.TASK, None))
        cls = get_factories(task, job_service)
        for caller in cls:
            try:
                instance = caller(args)
                return instance
            except Exception as e:
                logger.exception(f"[{caller.__name__}] {e}")
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
        return cls.create_from_dict(args)

    @property
    def algorithm(self) -> Algorithm:
        return self._instance

    @algorithm.setter
    def algorithm(self, instance: Algorithm) -> None:
        self._instance = instance

    def set_logger(
        self, path: Union[str, Path], stream_stdout: Optional[bool] = None
    ) -> None:
        set_logger(logging_path=path, stream_stdout=stream_stdout)

    def run(self) -> None:
        if self.algorithm is None:
            raise ValueError("No algorithm instance has been created.")

        self.algorithm.run()
