from utils._backports import StrEnum

from typing import Union, Literal, TypeVar
from dataclasses import dataclass, field
from enum import auto


class Tasks(StrEnum):
    TEXT_GENERATION = "llm"
    CLASSIFICATION = "image_classification"


class FactoryTypes(StrEnum):
    TEXT_GENERATION: Tasks = Tasks.TEXT_GENERATION
    VISION: Tasks = Tasks.CLASSIFICATION  # default


class FactoryArgumentKeys(StrEnum):
    ALGORITHM = "ALGORITHM"
    TASK = "TASK"
    ALGO_TYPE = "ALGO_TYPE"
    DATASET_NAME = "DATASET_NAME"
    USER_FOLDER = "USER_FOLDER"
    JOB_SERVICE = "JOB_SERVICE"
    CUSTOM_MODEL_PATH = "CUSTOM_MODEL_PATH"
    CACHE_PATH = "CACHE_PATH"
    DATA_PATH = "DATA_PATH"
    JOB_ID = "JOB_ID"
    MODEL_PATH = "MODEL_PATH"
    MODEL = "MODEL"
    TEXT_COLUMN = "TEXT_COLUMN"
    DATASET_SUBNAME = "DATASET_SUBNAME"
    SPLIT = "SPLIT"
    LOGGING_PATH = "LOGGING_PATH"
    FORMAT_STRING = "FORMAT_STRING"
    CUDA_ID = "CUDA_ID"


class AlgorithmKeys(StrEnum):
    AUTOAWQ = auto()
