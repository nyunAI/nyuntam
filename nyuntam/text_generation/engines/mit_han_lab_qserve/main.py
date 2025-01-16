# nyuntam
from nyuntam.algorithm import Algorithm

import os
import shutil
from pathlib import Path
from typing import Union, Optional

import logging

logger = logging.getLogger(__name__)

# script paths
CONVERTER = Path(__file__).parent / "ckpt_converter" / "checkpoint_converter.py"
assert CONVERTER.exists(), f"converter script not found at {CONVERTER}"

GENERATION = Path(__file__).parent / "scripts"
assert GENERATION.exists(), f"generation scripts not found at {GENERATION}"


class QServe(Algorithm):
    def __init__(self):
        NotImplementedError("Qserve is not available as an efficiency algorithm.")

    def compress_model(self):
        NotImplementedError("Qserve is not available as an efficiency algorithm.")

    @staticmethod
    def convert_checkpoint(
        model_path: Optional[Union[Path, str]],
        quant_path: Optional[Union[Path, str]],
        w_bit: str,
        group_size: str,
        output_path: Path,
        model_type: str = "LLaMa",
        device: str = "cpu",
    ) -> Path:
        """Convert a fake quantized LMQuant model to a real quantized model.

        Args:
            model_type (str): The model type to convert to. Default is "LLaMa".
            model_path (Path): The path to the original model checkpoint.
            quant_path (Path): The path to the quantized model checkpoint.
            w_bit (str): The bit width of the weights.
            group_size (int): The group size for the quantized model.
            output_path (Path): The path to save the converted model.

        Returns:
            output_path (Path): The path to the converted model."""

        assert model_type.lower() in [
            "llama",
        ], "We only support llama architecture for now."

        if not isinstance(model_path, Path):
            model_path = Path(model_path) if model_path else None

        if not isinstance(quant_path, Path):
            quant_path = Path(quant_path) if quant_path else None

        assert model_path and model_path.exists(), f"model not found at {model_path}"

        if quant_path:
            assert (
                quant_path.exists()
            ), f"quantized model checkpoint not found at {quant_path}"
            assert quant_path / "model.pt", f"model.pt not found at {quant_path}"
            assert quant_path / "scale.pt", f"scale.pt not found at {quant_path}"

        command = [
            "--model-type",
            model_type,
            "--model-path",
            str(model_path),
            "--quant-path",
            str(quant_path),
            "--w-bit",
            str(w_bit),
            "--group-size",
            str(group_size),
            "--output-path",
            str(output_path),
            "--device",
            device,
        ]
        exit_code = os.system(f"python {CONVERTER} {' '.join(command)}")
        assert (
            exit_code == 0
        ), f"checkpoint conversion failed with exit code {exit_code}"
        output_path = Path(output_path)
        return output_path

    @staticmethod
    def generate_run_script(
        output_path: Path = None,
    ):
        """Generate the run script for a quantized model with QServe."""
        # iteratively copy all files from the generation scripts folder to the output path
        for file in GENERATION.iterdir():
            dest = output_path / file.name
            if dest.exists():
                logger.warning(f"File {dest} already exists. Removing.")
                dest.unlink()
            shutil.copy(file, output_path / file.name)

        logger.info(
            f"Generated run script at {output_path}. Use the run.sh to run the model."
        )
