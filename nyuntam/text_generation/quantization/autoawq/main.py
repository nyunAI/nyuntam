from text_generation.core.job import LMJob

# nyuntam
from nyuntam.utils.device import CudaDeviceEnviron
from nyuntam.algorithm import TextGenerationAlgorithm

from typing import List, Dict
from awq import AutoAWQForCausalLM
from dataclasses import dataclass, field
from transformers import AutoTokenizer

import logging

logger = logging.getLogger(__name__)


@dataclass
class AutoAWQConfig:
    """Configuration for AutoAWQ"""

    # Model parameters
    model_path: str = field(default=None)
    """HF path to the model to be quantized"""

    output_path: str = field(default=None)
    """Path (of a directory) to save the quantized model"""

    # Quantization parameters
    zero_point: bool = field(default=True)
    """Whether to use zero point"""

    q_group_size: int = field(default=128)
    """Quantization group size"""

    w_bit: int = field(default=4)
    """Weight bitwidth"""

    version: str = field(default="GEMM")
    """Version of AutoAWQ. Can be GEMM or GEMV"""

    calib_data: str | List[str] = field(default="pileval")

    @classmethod
    def from_dict(cls, config: Dict = {}):
        """Create a configuration from a dictionary"""
        return cls(**config)

    def _validate(self):
        """Validate the configuration"""
        assert self.model_path is not None, "Model path must be provided"
        assert self.output_path is not None, "Output path must be provided"
        assert self.w_bit in [
            4,
        ], "Only 4-bit weight bitwidth is supported for now."  # TODO: Check what other bitwidths are supported
        assert self.version in [
            "GEMM",
            "GEMV",
        ], "Version must be GEMM or GEMV"  # TODO: Check what other versions are supported

    def __post_init__(self):
        """Post initialization"""
        self._validate()

    def to_dict(self):
        """Convert quant configuration to dictionary"""
        return {
            "zero_point": self.zero_point,
            "q_group_size": self.q_group_size,
            "w_bit": self.w_bit,
            "version": self.version,
        }


class _AutoAWQ:
    """AWQ (Activation Aware Quantization) via vLLM

    This class provides functionality for quantizing models using Activation Aware Quantization.

    Supported Models:
        - LLaMA-3: 8B/70B
        - LLaMA-2: 7B/13B/70B
        - LLaMA: 7B/13B/30B/65B
        - Mistral: 7B
        - Vicuna: 7B/13B
        - MPT: 7B/30B
        - Falcon: 7B/40B
        - OPT: 125m/1.3B/2.7B/6.7B/13B/30B
        - Bloom: 560m/3B/7B/
        - GPTJ: 6.7B
        - Aquila: 7B
        - Aquila2: 7B/34B"""

    def __init__(self, config: AutoAWQConfig):
        self.config: AutoAWQConfig = config

    def quantize(self) -> None:
        """Quantize the model"""
        factory_kwargs = {"low_cpu_mem_usage": True}
        factory_kwargs.update({"device_map": "auto"})

        logger.info(f"Quantization arguments: {self.config.to_dict()}")

        self.model = AutoAWQForCausalLM.from_pretrained(
            self.config.model_path, safetensors=True, **factory_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

        logger.info("Quantizing model...")
        self.model.quantize(
            self.tokenizer,
            quant_config=self.config.to_dict(),
            calib_data=self.config.calib_data,
        )
        self.model.save_quantized(
            str(self.config.output_path), shard_size="60GB"
        )  # shard size is higher to save a single model.safetensor file to be able to directly use in mlcllm
        self.tokenizer.save_pretrained(str(self.config.output_path))
        logger.info(f"Quantized model & tokenizer saved to {self.config.output_path}")


# ************
# Export Class
# ************


class AutoAWQ(TextGenerationAlgorithm):
    def __init__(self, job: LMJob, **kwargs):

        self.kw = kwargs
        self.quant_config: AutoAWQConfig = AutoAWQConfig.from_dict(
            {
                "model_path": job.model.model_path,
                "output_path": job.user_dir.output,
                "zero_point": self.kw.get("ZERO_POINT", True),
                "q_group_size": self.kw.get("Q_GROUP_SIZE", 128),
                "w_bit": self.kw.get("W_BIT", 4),
                "version": self.kw.get("VERSION", "GEMM"),
                "calib_data": job.dataset.load_caliberation_data(),
            }
        )
        self.quantizer = _AutoAWQ(self.quant_config)

    def compress_model(self):
        self.quantizer.quantize()
        return self.quantizer.model, None
