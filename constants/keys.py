from nyuntam.utils._backports import StrEnum
from enum import auto, Enum
from typing import Optional


class JobServices(Enum):
    ADAPT = auto()
    KOMPRESS = auto()

    @classmethod
    def get_service(cls, service: str):
        if service.lower() == "adapt":
            return cls.ADAPT
        elif service.lower() == "kompress":
            return cls.KOMPRESS
        else:
            raise ValueError(f"Invalid service: {service}")


class Task(StrEnum):

    @classmethod
    def create(cls, job_service: JobServices, task: str):
        if job_service == JobServices.KOMPRESS:
            return KompressTask.get_task_from_subclass(task)
        elif job_service == JobServices.ADAPT:
            return AdaptTasks.get_task(task)
        else:
            raise ValueError(f"Invalid job_service: {job_service}")

    @classmethod
    def get_task(cls, task: str):
        print(f"[{cls.__name__}] Task: {task}")
        return cls(task)

    @classmethod
    def has_subclass(cls):
        """Check if the class has any subclasses."""
        return len(cls.__subclasses__()) > 0

    @classmethod
    def get_task_from_subclass(cls, task: str):
        if not cls.has_subclass():
            return cls.get_task(task)
        else:
            for subclass in cls.__subclasses__():
                try:
                    if subclass.has_subclass():
                        return subclass.get_task_from_subclass(task)
                    else:
                        return subclass.get_task(task)
                except ValueError:
                    continue
        raise ValueError(f"Invalid task: {task}")


class KompressTask(Task):
    pass


class TextGenTasks(KompressTask):
    TEXT_GENERATION = "llm"


class VisionTasks(KompressTask):
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    IMAGE_SEGMENTATION = "image_segmentation"
    POSE_ESTIMATION = "pose_estimation"


class AdaptTasks(Task):
    TEXT_GENERATION = "text_generation"
    TEXT_CLASSIFICATION = "text_classification"
    SEQ_2_SEQ_TRANSLATION = "Seq2Seq_tasks"
    SEQ_2_SEQ_SUMMARIZATION = "Seq2Seq_tasks"
    QUESTION_ANSWERING = "question_answering"
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    IMAGE_SEGMENTATION = "image_segmentation"
    POSE_ESTIMATION = "pose_estimation"


class FactoryTypes(Enum):
    TEXT_GENERATION = auto()
    VISION = auto()
    ADAPT = auto()

    @classmethod
    def get_factory_type(
        cls, job_service: Optional[JobServices] = None, task: Optional[Task] = None
    ):
        assert (
            job_service is not None or task is not None
        ), "Either job_service or task must be provided"

        if job_service == JobServices.ADAPT:
            return cls.ADAPT
        elif job_service == JobServices.KOMPRESS:
            if isinstance(task, VisionTasks):
                return cls.VISION
            elif isinstance(task, TextGenTasks):
                return cls.TEXT_GENERATION
            else:
                raise ValueError(f"Invalid task: {task}")
        else:
            raise ValueError(f"Invalid job_service: {job_service}")


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
