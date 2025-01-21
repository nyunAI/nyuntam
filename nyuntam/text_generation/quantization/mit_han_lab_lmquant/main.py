from text_generation.core.job import LMJob
from text_generation.utils import (
    build_nested_dict,
    deep_update,
)
from text_generation.engines.mit_han_lab_qserve import QServe

# nyuntam
from nyuntam.algorithm import TextGenerationAlgorithm
from nyuntam.utils._backports import StrEnum


import yaml
from pathlib import Path
import logging
from typing import Dict, List, Dict, Set
import os

logger = logging.getLogger(__name__)


CONFIG_ROOT = Path(__file__).parent / "configs"


def get_cleaned_name(name: str) -> str:
    return name.replace("/", "_")


class QuantConfigHandler:

    class Config(StrEnum):
        LLM = "llm"  # default

        QOQ_GCHN = "gchn"  # qoq channelwise
        QOQ_G128 = "g128"  # qoq groupwise128

        SQ_DYNAMIC = "sq_dynamic"  # smooth quant dynamic
        SQ_STATIC = "sq_static"  # smooth quant static

        AWQ = "awq"
        GPTQ = "gptq"

        @classmethod
        def create(cls, quant_type: str) -> "Config":
            if not isinstance(quant_type, str):
                quant_type = str(quant_type)

            if quant_type == "default":
                return cls.LLM
            try:
                return cls(quant_type)
            except ValueError:
                raise ValueError(
                    f"Invalid quant_type: {quant_type} (Supported {', '.join(cls._config_to_path.keys() + ['default'])})"
                )

    _root = CONFIG_ROOT
    _config_to_path = {
        Config.QOQ_GCHN: _root / "qoq" / f"{Config.QOQ_GCHN}.yaml",
        Config.QOQ_G128: _root / "qoq" / f"{Config.QOQ_G128}.yaml",
        Config.SQ_DYNAMIC: _root / "smoothquant" / "dynamic.yaml",
        Config.SQ_STATIC: _root / "smoothquant" / "static.yaml",
        Config.AWQ: _root / "awq.yaml",
        Config.GPTQ: _root / "gptq.yaml",
        Config.LLM: _root / f"{Config.LLM}.yaml",
    }

    @classmethod
    def get_path(cls, config: "Config") -> Path:
        return cls._config_to_path[config]

    @classmethod
    def load_yaml_from_config(cls, config: Config) -> Dict:
        with open(cls.get_path(config), "r") as f:
            return yaml.safe_load(f)


class LMQuant(TextGenerationAlgorithm):

    cleanup_paths: Set[Path] = set()

    @staticmethod
    def cleanup():
        _removed = []
        for path in LMQuant.cleanup_paths:
            os.system(f"rm -rf {path}")
            logger.info(f"[LMQuant.cleanup] Removed: {path}")
            _removed.append(path)
        LMQuant.cleanup_paths = LMQuant.cleanup_paths - set(_removed)
        assert (
            len(LMQuant.cleanup_paths) == 0
        ), f"Failed to cleanup: {LMQuant.cleanup_paths}"

    @staticmethod
    def register_for_cleanup(path: Path):
        LMQuant.cleanup_paths.add(path)

    @property
    def output_path(self):
        for root, dirs, files in os.walk(self.job.user_dir.output):
            if "model.pt" in files and "scale.pt" in files:
                return Path(root)

    @property
    def converted_checkpoint_dir(self):
        for root, dirs, files in os.walk(self.job.user_dir.output):
            if "pytorch_model.bin" in files:
                return Path(root)

    def __init__(self, job: LMJob, **kwargs):

        self.job = job
        self.keep_scales = kwargs.pop("keep_scales", False)
        self.loads_with_qserve = kwargs.pop("loads_with_qserve", False)

        quant_type: QuantConfigHandler.Config = QuantConfigHandler.Config.create(
            kwargs.pop("quant_type", QuantConfigHandler.Config.LLM)
        )
        self.quant_type = quant_type

        default_qoq_config = QuantConfigHandler.load_yaml_from_config(quant_type)
        default_llm_config = QuantConfigHandler.load_yaml_from_config(
            QuantConfigHandler.Config.LLM
        )
        default_llm_config = deep_update(default_llm_config, default_qoq_config)
        model_name = get_cleaned_name(job.model.model_name.lower())
        model_name = model_name.replace(job.model.size, "").replace(
            job.model.size.lower(), ""
        )
        model_name += f"-{job.model.size.lower()}"
        _updated_mappings = {
            "model.name": model_name,
            "model.local_path": str(job.model.model_path.absolute().resolve()),
            "calib.dataset_path": str(
                job.dataset.dataset_name_or_path.absolute().resolve()
            ),
            "calib.local_dataset_path": str(
                job.dataset.dataset_name_or_path.absolute().resolve()
            ),
            "calib.data": get_cleaned_name(kwargs.get("DATASET_NAME", "wikitext")),
            "eval.output_dirname": "evals",
            "eval.output_root": str(job.user_dir.output.absolute().resolve()),
            "eval.cache_root": str(job.user_dir.cache.absolute().resolve()),
            "eval.attach_timestamp": False,
            "calib.cache_root": str(job.user_dir.cache.absolute().resolve()),
            "output_dirname": "",
        }
        kwargs.update(_updated_mappings)
        expanded_kwargs = build_nested_dict(kwargs)
        llm_config = deep_update(default_llm_config, expanded_kwargs)

        # create the config file
        self.runner_file = job.user_dir.output / "runner_config.yaml"
        with open(self.runner_file, "w") as f:
            yaml.dump(llm_config, f)

        # temp data path for custom dataset
        temp_data_path = job.dataset.to_json_file(job.user_dir.output / "temp")
        os.environ["KOMPRESS_DATA_PATH"] = str(temp_data_path)
        LMQuant.register_for_cleanup(Path(temp_data_path))

    def convert_checkpoint(self):
        with open(self.output_path / "config.yaml", "r") as f:
            config_data = yaml.safe_load(f)

        return QServe.convert_checkpoint(
            model_type="LLaMa",
            model_path=str(self.job.model.model_path.absolute().resolve()),
            quant_path=str(self.output_path),
            w_bit=4 if "int4" in config_data["quant"]["wgts"]["dtype"] else 8,
            group_size=(
                128 if self.quant_type == QuantConfigHandler.Config.QOQ_G128 else -1
            ),
            output_path=self.job.user_dir.output,
            device="cpu",
        )

    def generate_run_script(self):
        """Generate the run script for the quantized model with QServe."""
        QServe.generate_run_script(self.converted_checkpoint_dir)

    def quantize(self):
        exit_code = os.system(f"python -m lmquant.llm.run {self.runner_file}")
        if exit_code != 0:
            raise Exception("Failed to compress model.")

        LMQuant.register_for_cleanup(self.runner_file)
        if not self.keep_scales:
            LMQuant.register_for_cleanup(
                self.job.user_dir.output / "llm"
            )  # top-most parent to lmquant's scale.pt and model.pt

    def compress_model(self):
        # 1. generate fake quants
        self.quantize()

        if self.loads_with_qserve:
            # 2. use qserve to convert the checkpoints
            self.convert_checkpoint()

            # 3. generate run script (uses qserve)
            self.generate_run_script()

        # cleanup unwanted files (quant configs, scales, intermediary logs, etc.)
        LMQuant.cleanup()
        return __name__, None
