from text_generation.core.job import LMJob
from text_generation.quantization.aqlm.config import (
    AQLMConfig,
    CalibrationConfig,
    FineTuneConfig,
    ConversionConfig,
)
from text_generation.quantization.aqlm.utils import (
    caliberate_model,
    finetune_quantized,
    convert_to_hf,
    get_rank,
)
from text_generation.utils import create_instance, log_dict

# nyuntam
from nyuntam.algorithm import TextGenerationAlgorithm

from dataclasses import asdict
import logging

logger = logging.getLogger(__name__)


class AQLM(TextGenerationAlgorithm):

    def __init__(self, job: LMJob, **kwargs):
        rank = get_rank()
        self.job = job
        self.config: AQLMConfig = AQLMConfig(
            save_intermediate_results=kwargs.get("save_intermediate_results", False),
            calibration_config=create_instance(
                CalibrationConfig, kwargs.get("calibration_config", {})
            ),
            finetune_config=create_instance(
                FineTuneConfig, kwargs.get("finetune_config", {})
            ),
            conversion_config=create_instance(
                ConversionConfig, kwargs.get("conversion_config", {})
            ),
        )
        tmp_dir = self.job.user_dir.get_tmp_output_dir(
            register_for_cleanup=not self.config.save_intermediate_results
        )

        calib_outputs = tmp_dir / "caliberation"
        dataset_outputs = tmp_dir / "tokenized_dataset"
        finetune_outputs = tmp_dir / "finetune"
        conversion_outputs = str(self.job.user_dir.output)

        # caliberation config
        self.config.calibration_config.model_path = str(job.model.model_path)
        self.config.calibration_config.job_dataset = job.dataset
        self.config.calibration_config.save = str(calib_outputs)

        # finetune config
        self.config.finetune_config.base_model = str(job.model.model_path)
        self.config.finetune_config.quantized_model = (
            self.config.calibration_config.save
        )
        self.config.finetune_config.dataset_name = str(dataset_outputs)
        self.config.finetune_config.save = str(finetune_outputs)

        # conversion config
        self.config.conversion_config.base_model = str(job.model.model_path)
        self.config.conversion_config.quantized_model = str(job.model.model_path)
        self.config.conversion_config.pv_fsdp_dir = self.config.finetune_config.save
        self.config.conversion_config.save = conversion_outputs

        if rank == 0:
            log_dict(
                {
                    **asdict(self.config),
                    "overwrite_or_run_caliberation": self.config.overwrite_or_run_caliberation,
                    "overwrite_or_run_conversion": self.config.overwrite_or_run_conversion,
                    "overwrite_or_run_dataset_tokenization": self.config.overwrite_or_run_dataset_tokenization,
                    "overwrite_or_run_finetune": self.config.overwrite_or_run_finetune,
                    "overwrite_or_run_all": self.config.overwrite_or_run_all,
                },
                prefix="AQLMConfig.",
            )
            if self.config.overwrite:
                logger.warning(
                    "Overwrite is enabled. Existing files will be overwritten for any intermediate results."
                )

    def compress_model(self):
        # Caliberate the model
        caliberate_model(self.config)

        # Tokenize the dataset
        self.job.dataset.tokenize_aqlm(
            self.config, self.config.finetune_config.dataset_name
        )

        # Finetune the quantized model
        finetune_quantized(self.config)

        # Convert the model to HF format
        convert_to_hf(self.config)

        self.job.user_dir.cleanup()
