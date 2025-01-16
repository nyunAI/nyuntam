from text_generation.core.model import LanguageModel
from text_generation.core.dataset import Dataset

# nyuntam
from nyuntam.utils.device import CudaDeviceEnviron

from dataclasses import dataclass, field, fields
from pathlib import Path

from typing import Set
from pathlib import Path
import os
from logging import getLogger

logger = getLogger(__name__)


@dataclass
class UserDir:
    job_id: int = field(default=None, metadata={"help": "ID of the Job."})
    root: Path = field(
        default="abc", metadata={"help": "Root directory for the Job. (User folder)"}
    )
    job_service: str = field(
        default=None, metadata={"help": "Service to be used for the Job."}
    )
    cache: Path = field(default=None, metadata={"help": "Cache directory for the Job."})
    output: Path = field(
        default=None, metadata={"help": "Output directory for the Job."}
    )
    logs: Path = field(default=None, metadata={"help": "Logs directory for the Job."})
    models: Path = field(
        default=None, metadata={"help": "Models directory for the Job."}
    )
    datasets: Path = field(
        default=None, metadata={"help": "Datasets directory for the Job."}
    )
    jobs: Path = field(default=None, metadata={"help": "Jobs directory for the Job."})

    __cleanup_paths: Set[Path] = field(default_factory=set, init=False)

    def cleanup(self):
        """A job cleanup method to remove intermediary files and directories registered for cleanup."""
        _removed = []
        for path in self.__cleanup_paths:
            os.system(f"rm -rf {path}")
            logger.info(f"[{self.__name__}.cleanup] Removed: {path}")
            _removed.append(path)
        self.__cleanup_paths = self.__cleanup_paths - set(_removed)
        assert (
            len(self.__cleanup_paths) == 0
        ), f"Failed to cleanup: {self.__cleanup_paths}"

    def register_for_cleanup(self, path: Path):
        """Register a path for cleanup."""
        self.__cleanup_paths.add(path)

    def __post_init__(self):
        if self.root is None:
            raise ValueError("Root directory for the Job is required.")
        if self.job_id is None:
            raise ValueError("ID of the Job is required.")
        if self.job_service is None:
            raise ValueError("Service to be used for the Job is required.")

        if self.cache is None:
            self.cache = self.root / ".cache"
        if self.output is None:
            self.output = self.root / "jobs" / str(self.job_service) / str(self.job_id)
        if self.logs is None:
            self.logs = self.root / "logs" / str(self.job_service) / str(self.job_id)
        if self.models is None:
            self.models = self.root / "models"
        if self.datasets is None:
            self.datasets = self.root / "datasets"

        paths = [self.cache, self.output, self.logs, self.models, self.datasets]
        for path in paths:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)

    def _get_tmp_dir(self, root: Path, register_for_cleanup=True) -> Path:
        tmp_dir = root / "tmp"
        if not tmp_dir.exists():
            tmp_dir.mkdir(parents=True, exist_ok=True)
        if register_for_cleanup:
            self.register_for_cleanup(tmp_dir)
        return tmp_dir

    def get_tmp_output_dir(self, register_for_cleanup=True) -> Path:
        """Create a temporary directory at the Job output."""
        return self._get_tmp_dir(self.output, register_for_cleanup)


@dataclass
class Job:

    ## Ideally a Job should be the entrypoint to the package & have the following fields (sourced from sample.json):
    # id: int = field(default=None, metadata={"help": "ID of the Job."})
    # job_service: str = field(default=None, metadata={"help": "Service to be used for the Job."})
    # model: KompressModel = field(default=None, metadata={"help": "Model to be used for the Job."})
    # dataset: KompressDataset = field(default=None, metadata={"help": "Dataset to be used for the Job."})
    # method: str = field(default=None, metadata={"help": "Method to be used for the Job."})
    # method_hyperparameters: Hyperparameters = field(default=None, metadata={"help": "Hyperparameters for the method."})
    # job_status: Status = field(default=None, metadata={"help": "Status of the Job."})
    # parent_folder: str = field(default=None, metadata={"help": "Parent folder for the Job."})

    # TODO: modify the structure of the Job class

    model: object = field(
        default=None, metadata={"help": "Model to be used for the Job."}
    )
    dataset: object = field(
        default=None, metadata={"help": "Dataset to be used for the Job."}
    )
    user_dir: UserDir = field(
        default=None, metadata={"help": "Directory structure for the Job."}
    )

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in fields(cls)})

    @classmethod
    def from_path(cls, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"File {path} not found.")
        if path.is_dir():
            raise ValueError(f"Path {path} is a directory, not a file.")

        if path.suffix == ".json":
            return cls.from_json(path)
        elif path.suffix == ".yaml":
            return cls.from_yaml(path)
        else:
            raise ValueError(f"File format {path.suffix} not supported.")

    @classmethod
    def from_json(cls, json_path: Path):
        import json

        if not json_path.exists() or not json_path.is_file():
            raise Exception(f"Path {json_path} not found or is not a file.")
        with open(json_path, "r") as f:
            json = json.load(f)
        return cls.from_dict(json)

    @classmethod
    def from_yaml(cls, yaml_path: Path):
        import yaml

        if not yaml_path.exists() or not yaml_path.is_file():
            raise Exception(f"Path {yaml_path} not found or is not a file.")
        with open(yaml_path, "r") as f:
            yaml = yaml.safe_load(f)
        return cls.from_dict(yaml)


@dataclass
class LMJob(Job):
    model: LanguageModel = field(
        default=None, metadata={"help": "Language Model to be used for the Job."}
    )
    dataset: Dataset = field(
        default=None, metadata={"help": "Dataset to be used for the Job."}
    )
    environment: CudaDeviceEnviron = field(
        default=None, metadata={"help": "Device Environment"}
    )  # TODO: make this adaptive to different device types (cuda, rom, metal, etc.)
