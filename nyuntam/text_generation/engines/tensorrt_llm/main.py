from text_generation.core.job import LMJob

# nyuntam
from nyuntam.algorithm import TextGenerationAlgorithm

import gc
import os
import json
import torch
import shutil
import logging
import subprocess
import tensorrt_llm
from enum import Enum
from pathlib import Path
from typing import Dict, Set


def free(times: int = 2):
    for _ in range(times):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


os.environ["TLLM_LOG_LEVEL"] = "INFO"
tensorrt_llm.logger.set_level("info")

logger = logging.getLogger(__name__)

KOMPRESS_ENVIRON_DATASET_JSON_PATH = "KOMPRESS_ENVIRON_DATASET_JSON_PATH"

# path to trtllm inside docker
THIS = Path(__file__)
TENSORRTLLM = THIS.parent / "TensorRT-LLM"
SCRIPTS = TENSORRTLLM / "examples"


# commands / scripts
def CONVERT_CHECKPOINT(model: "Model"):
    return SCRIPTS / str(model) / "convert_checkpoint.py"


QUANTIZE = SCRIPTS / "quantization" / "quantize.py"
BUILD = "trtllm-build"
RUN = SCRIPTS / "run.py"


def log_subprocess_output(pipe):
    for line in pipe.readlines():
        log_content = line.strip()
        if log_content != "":
            logger.info(log_content)


def run_cmd(cmd, process: str, register_proc):
    print("running...", " ".join(cmd))
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        universal_newlines=True,
    ) as proc:
        logger.info(process + "...")
        log_subprocess_output(proc.stdout)
        register_proc(proc)
    free()


class Quant(Enum):
    # via QUANTIZE
    INT4AWQ = "int4_awq"
    FP8 = "fp8"
    # via CONVERT_CHECKPOINT
    SMOOTH_QUANT = "smoothquant"
    INT8 = "int8"

    def __str__(self) -> str:
        return self.value


class Model(Enum):
    # TODO: Test all models
    BAICHUAN = "baichuan"
    BERT = "bert"
    BLIP2 = "blip2"
    BLOOM = "bloom"
    CHATGLM = "chatglm"
    ENC_DEC = "enc_dec"
    FALCON = "falcon"
    GEMMA = "gemma"  # ✅ Tested
    GPT = "gpt"
    GPTJ = "gptj"
    GPTNEOX = "gptneox"
    INTERNLM = "internlm"
    LLAMA = "llama"  # ✅ Tested
    MAMBA = "mamba"
    MEDUSA = "medusa"
    MPT = "mpt"
    MULTIMODAL = "multimodal"
    OPT = "opt"
    PHI = "phi"  # ✅ Tested
    WHISPER = "whisper"

    @classmethod
    def for_mixed_support(cls, model_type: str):
        """This method returns the model key for mixed implementations of models.
        For eg. Mistral & Mixtral are supported via Llama implementation
        """

        if model_type in {"mistral", "mixtral"}:  # ✅ Tested
            return cls(cls.LLAMA)
        else:
            logger.warn("Unknown model type, loading as llama.")
            return cls(cls.LLAMA)

    def __str__(self) -> str:
        return self.value


QUANT_MODEL_MAP: Dict[Model, Set[Quant]] = {
    # llama, vicuna
    Model.LLAMA: {Quant.INT4AWQ, Quant.SMOOTH_QUANT, Quant.FP8, Quant.INT8},
    Model.PHI: set(),
    Model.GEMMA: {Quant.INT4AWQ, Quant.FP8, Quant.INT8},
}


def load_model_type_from_config_path(config_path: Path):
    if not config_path.is_file() or not config_path.exists():
        raise FileNotFoundError(f"Path {config_path} is not file or does not exists.")

    with open(config_path, "r") as f:
        config = json.load(f)
        return config["model_type"]


class TensorRTLLM(TextGenerationAlgorithm):

    __procs = list()

    def __init__(self, job: LMJob, **kwargs):
        self.job = job
        self.kwargs = kwargs
        self.to_quantize = kwargs.get("to_quantize", True)
        self.quant_method = Quant(kwargs.get("quant_method", "int4_awq"))

        model_input_dir = job.model.model_path

        self.model_type = load_model_type_from_config_path(
            model_input_dir / "config.json"
        )
        """model type from config.json"""

        try:
            self.model = Model(self.model_type)
        except:
            self.model = Model.for_mixed_support(self.model_type)

        self.quantizer = None

        self.quant_keys = {}
        self.build_keys = {}
        self.convert_checkpoint_keys = {}

        # dump dataset to be used by tensorr_tllm and set environ
        datset_json_path = self.job.dataset.to_json_file(
            self.job.user_dir.cache / "temp"
        )
        os.environ[KOMPRESS_ENVIRON_DATASET_JSON_PATH] = str(
            datset_json_path.absolute()
        )

    def validate_quant_support(self, model: Model, quant_method: Quant):
        """This method returns a boolean whether a quant_method is supported for a model type"""
        if model not in QUANT_MODEL_MAP.keys():
            return False

        # diff_set for cases where mixed support is loaded e.g. mistral, mixtral loaded via llama
        diff_set: Set = set()
        if self.model_type in {"mistral", "mixtral"}:
            # mistral doesn't support sq.
            # TODO: Add support if required
            diff_set = {Quant.SMOOTH_QUANT}

        return quant_method in QUANT_MODEL_MAP[model].difference(diff_set)

    def quantize(self, quant_method: Quant, model: Model):

        quant_output_dir = str(
            self.job.user_dir.output / f"tllm_checkpoint_{model}_{quant_method}"
        )
        assert self.validate_quant_support(
            model=model, quant_method=quant_method
        ), f"Unsupported quant_method: `{quant_method}`. Please check the documentation for supported quantizations."

        OUTPUT_DIR_KEY = "output_dir"
        MODEL_DIR_KEY = "--model_dir"

        # ================================================================
        #                       MODEL SPECIFIC KEYS
        # ================================================================

        if model == Model.LLAMA:
            if quant_method == Quant.INT4AWQ:
                self.quant_keys.update(
                    {
                        "qformat": str(quant_method),
                        "dtype": self.kwargs.get("dtype", "float16"),
                        "awq_block_size": 128,
                        "calib_size": self.kwargs.get("calib_size", 32),
                        "output_dir": quant_output_dir,
                    }
                )
                self.build_keys.update(
                    {
                        "checkpoint_dir": quant_output_dir,
                        "gemm_plugin": self.kwargs.get("dtype", "float16"),
                    }
                )

            elif quant_method == Quant.FP8:
                self.quant_keys.update(
                    {
                        "qformat": str(quant_method),
                        "dtype": self.kwargs.get("dtype", "float16"),
                        "calib_size": self.kwargs.get("calib_size", 32),
                        "output_dir": quant_output_dir,
                    }
                )
                self.build_keys.update({"checkpoint_dir": quant_output_dir})

            elif quant_method == Quant.SMOOTH_QUANT:
                self.convert_checkpoint_keys.update(
                    {
                        "smoothquant": self.kwargs.get("smoothquant", 0.5),
                        "dtype": self.kwargs.get("dtype", "float16"),
                        "per_token": self.kwargs.get("per_token", None),
                        "per_channel": self.kwargs.get("per_channel", None),
                        "output_dir": quant_output_dir,
                    }
                )
                self.build_keys.update(
                    {
                        "checkpoint_dir": quant_output_dir,
                        "gemm_plugin": self.kwargs.get("dtype", "float16"),
                    }
                )

            elif quant_method == Quant.INT8:
                self.convert_checkpoint_keys.update(
                    {
                        "use_weight_only": True,
                        "dtype": self.kwargs.get("dtype", "float16"),
                        "weight_only_precision": str(quant_method),
                        "output_dir": quant_output_dir,
                    }
                )
                self.build_keys.update(
                    {
                        "checkpoint_dir": quant_output_dir,
                        "gemm_plugin": self.kwargs.get("dtype", "float16"),
                    }
                )

        elif model == Model.PHI:
            pass

        elif model == Model.GEMMA:

            OUTPUT_DIR_KEY = "output-model-dir"
            MODEL_DIR_KEY = "--model-dir"

            if quant_method == Quant.INT4AWQ:
                self.quant_keys.update(
                    {
                        "qformat": str(quant_method),
                        "dtype": self.kwargs.get("dtype", "float16"),
                        "awq_block_size": 128,
                        "calib_size": self.kwargs.get("calib_size", 32),
                        "output_dir": quant_output_dir,
                    }
                )
                self.build_keys.update(
                    {
                        "checkpoint_dir": quant_output_dir,
                        "gemm_plugin": self.kwargs.get("dtype", "float16"),
                    }
                )

            elif quant_method == Quant.FP8:
                self.quant_keys.update(
                    {
                        "qformat": str(quant_method),
                        "dtype": self.kwargs.get("dtype", "float16"),
                        "calib_size": self.kwargs.get("calib_size", 32),
                        "output_dir": quant_output_dir,
                    }
                )
                self.build_keys.update({"checkpoint_dir": quant_output_dir})
            elif quant_method == Quant.INT8:
                self.convert_checkpoint_keys.update(
                    {
                        "ckpt-type": "hf",
                        "dtype": self.kwargs.get("dtype", "float16"),
                        "use-weight-only-with-precision": str(quant_method),
                        OUTPUT_DIR_KEY: quant_output_dir,
                    }
                )
                self.build_keys.update(
                    {
                        "checkpoint_dir": quant_output_dir,
                        "gemm_plugin": self.kwargs.get("dtype", "float16"),
                    }
                )

        # ================================================================
        #                            QUANTIZE
        # ================================================================

        def is_empty(keys):
            return len(keys.keys()) == 0

        # QUANTIZE
        if not is_empty(self.quant_keys):
            quant_cmd = [
                "python",
                str(QUANTIZE),
                "--model_dir",
                str(self.job.model.model_path),
            ]

            self.quant_keys.update(
                {
                    "max_seq_length": self.kwargs.get("max_seq_length", 2048),
                    "batch_size": self.kwargs.get("batch_size", 1),
                }
            )

            for key, value in self.quant_keys.items():
                if value is not None and value:
                    if type(value) == type(True) and value == True:
                        quant_cmd.extend([f"--{key}"])
                    else:
                        quant_cmd.extend([f"--{key}", f"{value}"])
            run_cmd(quant_cmd, "quantizing", self.add_to_procs)

        # quant with CONVERT_CHECKPOINT
        elif not is_empty(self.convert_checkpoint_keys):
            convert_ckpt_cmd = [
                "python",
                str(CONVERT_CHECKPOINT(model=model)),
                MODEL_DIR_KEY,
                str(self.job.model.model_path),
            ]
            for key, value in self.convert_checkpoint_keys.items():
                if value is not None and value:
                    if type(value) == type(True) and value == True:
                        convert_ckpt_cmd.extend([f"--{key}"])
                    else:
                        convert_ckpt_cmd.extend([f"--{key}", f"{value}"])
            run_cmd(convert_ckpt_cmd, "converting quant checkpoints", self.add_to_procs)

    def convert_checkpoints(self, model: Model):
        # convert checkpoints for non-quant usecases

        OUTPUT_DIR_KEY = "output_dir"
        MODEL_DIR_KEY = "--model_dir"

        if model == Model.GEMMA:
            OUTPUT_DIR_KEY = "output-model-dir"
            MODEL_DIR_KEY = "--model-dir"
        ckpt_output_dir = str(self.job.user_dir.output / f"tllm_checkpoint_{model}")
        self.convert_checkpoint_keys.update(
            {
                "dtype": self.kwargs.get("dtype", "float16"),
                OUTPUT_DIR_KEY: ckpt_output_dir,
            }
        )

        if model == Model.LLAMA:
            self.convert_checkpoint_keys.update(
                {
                    "tp_size": self.job.environment.num_device,
                    "per_channel": self.kwargs.get("per_channel", False),
                    "per_token": self.kwargs.get("per_token", False),
                    "int8_kv_cache": self.kwargs.get("int8_kv_cache", False),
                    "per_group": self.kwargs.get("per_group", False),
                    "hidden_act": self.kwargs.get(
                        "hidden_act", "silu"
                    ),  # erroneous if model generated with transformers <= v4.36 ( TODO: validate the version that has a fix )
                    "rotary_base": self.kwargs.get("rotary_base", 10000.0),
                }
            )
        elif model == Model.GEMMA:
            self.convert_checkpoint_keys.update({"ckpt-type": "hf"})

        self.build_keys.update(
            {
                "checkpoint_dir": ckpt_output_dir,
                "gemm_plugin": self.kwargs.get("dtype", "float16"),
            }
        )

        if self.model_type in {"mistral"}:
            self.build_keys.update(
                {"max_input_len": self.kwargs.get("max_input_len", 32256)}
            )

        convert_ckpt_cmd = [
            "python",
            str(CONVERT_CHECKPOINT(model)),
            MODEL_DIR_KEY,
            str(self.job.model.model_path),
        ]

        for key, value in self.convert_checkpoint_keys.items():
            if value is not None and value:
                if type(value) == type(True) and value == True:
                    convert_ckpt_cmd.extend([f"--{key}"])
                else:
                    convert_ckpt_cmd.extend([f"--{key}", f"{value}"])
        run_cmd(convert_ckpt_cmd, "converting checkpoints", self.add_to_procs)

    def compress_model(self):

        if self.to_quantize:
            # quantizes and converts checkpoints
            self.quantize(self.quant_method, self.model)
        else:
            self.convert_checkpoints(self.model)

        build_cmd = [BUILD, "--output_dir", str(self.job.user_dir.output)]
        self.build_keys.update(
            {
                "workers": self.job.environment.num_device,
                "gpus_per_node": self.job.environment.num_device,
                "tp_size": self.job.environment.num_device,
                "max_batch_size": self.kwargs.get("max_batch_size", 1),
                "max_input_len": self.kwargs.get("max_input_len", 1024),
                "max_output_len": self.kwargs.get("max_output_len", 1024),
                "max_beam_width": self.kwargs.get("max_beam_width", 1),
                "max_num_tokens": self.kwargs.get("max_num_tokens", None),
                "max_prompt_embedding_table_size": self.kwargs.get(
                    "max_prompt_embedding_table_size", 0
                ),
                "use_fused_mlp": self.kwargs.get("use_fused_mlp", False),
            }
        )

        for key, value in self.build_keys.items():
            if value is not None and value:
                if type(value) == type(True) and value == True:
                    build_cmd.extend([f"--{key}"])
                else:
                    build_cmd.extend([f"--{key}", f"{value}"])
        run_cmd(build_cmd, "building engine", self.add_to_procs)

        if any([p.returncode != 0 for p in self.__procs]):
            raise ChildProcessError("Subprocess failed. Check logs for error trace.")

        self.export_scripts()

        return None, None

    def export_scripts(self):
        # exports run scripts for the built engines
        logger.info("exporting scripts")

        run_script_path = self.job.user_dir.output / "run.py"
        shutil.copy(RUN, run_script_path)

        logger.info("=" * 50 + " Run " + "=" * 50)
        logger.info("To run the model using the engines generated use,")
        logger.info(
            f"python {run_script_path.absolute()} --max_output_len 1000 \
--tokenizer_dir {self.job.model.model_path.absolute()} \
--engine_dir {self.job.user_dir.output.absolute()}"
        )
        logger.info(
            "ref - https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/run.py"
        )
        logger.info("=" * 100)

    def add_to_procs(self, proc):
        self.__procs.append(proc)
