from nyuntam.utils._backports import StrEnum

from dataclasses import dataclass, field
from enum import auto


class JobServices(StrEnum):
    ADAPT = "Adapt"
    KOMPRESS = "Kompress"


class KompressTasks(StrEnum):
    TEXT_GENERATION = "llm"
    CLASSIFICATION = "image_classification"


class AdaptTasks(StrEnum):
    TEXT_GENERATION = "text_generation"
    TEXT_CLASSIFICATION = "text_classification"
    SEQ_2_SEQ_TRANSLATION = "Seq2Seq_tasks"
    SEQ_2_SEQ_SUMMARIZATION = "Seq2Seq_tasks"
    QUESTION_ANSWERING = "question_answering"
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    IMAGE_SEGMENTATION = "image_segmentation"
    POSE_ESTIMATION = "pose_estimation"


class FactoryTypes(StrEnum):
    TEXT_GENERATION: KompressTasks = KompressTasks.TEXT_GENERATION
    VISION: KompressTasks = KompressTasks.CLASSIFICATION  # default
    ADAPT: JobServices = JobServices.ADAPT


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
