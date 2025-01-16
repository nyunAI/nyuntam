from text_generation import __getattr__ as get_algorithm
from text_generation.core.model import LanguageModel
from text_generation.core.job import LMJob, UserDir
from text_generation.core.dataset import Dataset

# nyuntam
from nyuntam.factory import Factory as BaseFactory, FactoryTypes
from nyuntam.constants.keys import FactoryArgumentKeys
from nyuntam.utils.device import CudaDeviceEnviron
from nyuntam.utils.dtype import get_dtype_from_string
from nyuntam.algorithm import TextGenerationAlgorithm


from pathlib import Path
from typing import Optional


class Factory(BaseFactory):
    """Factory class for creating text-generation algorithms."""

    _type = FactoryTypes.TEXT_GENERATION

    def get_algorithm(self, name: str) -> TextGenerationAlgorithm:
        return get_algorithm(name)

    def __init__(self, args: dict) -> BaseFactory:
        """Initialize LM Job and sets the compressor object."""

        # TODO: make it cleaner and more modular. Distribute constants.
        super().__init__(args)

        # paths
        cache_path = args.get(FactoryArgumentKeys.CACHE_PATH, None)
        output_path = args.get(FactoryArgumentKeys.MODEL_PATH, None)
        logs_path = args.get(FactoryArgumentKeys.LOGGING_PATH, None)

        cache_path = Path(cache_path) if cache_path else None
        output_path = Path(output_path) if output_path else None
        logs_path = Path(logs_path) if logs_path else None

        user_dir = UserDir(
            job_id=args.get(FactoryArgumentKeys.JOB_ID),
            root=Path(args.get(FactoryArgumentKeys.USER_FOLDER)),
            job_service=args.get(FactoryArgumentKeys.JOB_SERVICE),
            cache=cache_path,
            output=output_path,
            logs=logs_path,
        )

        self.set_logger(user_dir.logs)

        model: LanguageModel = None
        model_name: str = args.get(FactoryArgumentKeys.MODEL, "huggyllama/llama-7b")

        dataset: Dataset = None
        split: str = args.get(FactoryArgumentKeys.SPLIT, "train")
        text_column: str = args.get(FactoryArgumentKeys.TEXT_COLUMN, "text")
        format_string: str = args.get(FactoryArgumentKeys.FORMAT_STRING, None)
        dataset_name: Optional[str] = args.get(FactoryArgumentKeys.DATASET_NAME, None)
        dataset_subname: Optional[str] = args.get(
            FactoryArgumentKeys.DATASET_SUBNAME, None
        )

        # custom paths
        dataset_path: Optional[str] = args.get(FactoryArgumentKeys.DATA_PATH, None)
        custom_model_path: Optional[str] = args.get(
            FactoryArgumentKeys.CUSTOM_MODEL_PATH, None
        )

        custom_model_path = Path(custom_model_path) if custom_model_path else None
        dataset_path = Path(dataset_path) if dataset_path else None

        # set visible cuda devices
        cuda_device_str = args.get(FactoryArgumentKeys.CUDA_ID, "0")
        device_environ = CudaDeviceEnviron(cuda_device_ids_str=cuda_device_str)

        dtype = get_dtype_from_string(
            args.get(args.get(FactoryArgumentKeys.TASK))
            .get(args.get(FactoryArgumentKeys.ALGORITHM, "AutoAWQ"))
            .get("dtype", "float16")
        )
        method = args.get(FactoryArgumentKeys.ALGORITHM, "AutoAWQ")

        model = LanguageModel.from_model_path_or_name(
            model_name=model_name,
            custom_model_path=custom_model_path,
            dtype=dtype,
            save_dir=user_dir.models / model_name,
            cache_dir=user_dir.cache,
            patch_phi3=method == "LMQuant",
        )

        dataset = Dataset.from_name_or_path(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            dataset_subset=dataset_subname,
            save_dir=user_dir.datasets / dataset_name,
            cache_dir=user_dir.cache,
            format_string=format_string,
            text_column=text_column,
            split=split,
        )

        job = LMJob(
            model=model, dataset=dataset, user_dir=user_dir, environment=device_environ
        )

        # flatten args
        kw = {}
        for k in args.keys():
            if type(args[k]) != type({}):
                kw.update({k: args[k]})
        kw.update(args["llm"][method])
        self.algorithm = self.get_algorithm(name=method)(job=job, **kw)
