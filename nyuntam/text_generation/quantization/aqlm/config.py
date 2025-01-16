from text_generation.core.dataset import Dataset

from dataclasses import dataclass, field
from typing import Optional, Union, Iterable
from functools import cached_property
from pathlib import Path
import os

# ============ Configs ============


@dataclass
class CalibrationConfig:
    model_path: str = "model_path"
    dataset: str = "dataset"
    save: Optional[Union[str, Path]] = None

    dataset_name: Optional[str] = None
    job_dataset: Optional[Dataset] = None

    attn_implementation: Optional[str] = None
    beam_size: int = 1
    codebook_value_nbits: int = 16
    codebook_value_num_groups: int = 1
    devices: Optional[Iterable] = None
    dtype: str = "auto"
    exp_name: Optional[str] = None
    finetune_adam_beta1: float = 0.90
    finetune_adam_beta2: float = 0.999
    finetune_batch_size: int = 16  # 1 gpu
    finetune_early_stop: int = 3
    finetune_keep_best: bool = True
    finetune_lr: float = 1e-4
    finetune_max_epochs: int = 25
    in_group_size: int = 8
    init_max_iter: int = 100
    init_max_points_per_centroid: Optional[int] = None
    load: Optional[Union[str, Path]] = None
    local_batch_size: Optional[int] = 1  # 1 gpu
    lr: float = 0.0001
    max_epochs: int = 100
    mix_compression: bool = False
    model_seqlen: int = 4096
    nbits_per_codebook: int = 16
    new_eval: bool = False
    no_quant: bool = False
    nsamples: Optional[int] = 2048
    num_codebooks: int = 1
    offload_activations: bool = True
    on_save: Optional[str] = None
    out_group_size: int = 1
    print_frequency: int = 10
    relative_mse_tolerance: Optional[float] = 0.01
    resume: bool = False
    scale_nbits: int = 0
    seed: int = 0
    skip_out_loss: bool = False
    steps_per_epoch: int = 100
    true_sequential: bool = False
    trust_remote_code: bool = True
    use_checkpointing: bool = False
    use_faiss: bool = False
    use_fast_tokenizer: bool = False
    val_size: int = 256
    wandb: bool = False

    def __post_init__(self):
        assert self.attn_implementation in [
            None,
            "eager",
            "flash_attention_2",
            "sdpa",
        ], f'Invalid attn_implementation: {self.attn_implementation}. Allowed values: [None, "eager", "flash_attention_2", "sdpa"]]'
        assert self.dtype in [
            "auto",
            "float16",
            "float32",
            "bfloat16",
        ], f'Invalid dtype: {self.dtype}. Allowed values: ["auto", "float16", "float32", "bfloat16"]'
        if self.val_size and self.nsamples:
            assert (
                self.val_size < self.nsamples
            ), "Number of validation set must be smaller than train + val"


@dataclass
class FineTuneConfig:

    ### model
    base_model: str = "base_model"
    block_type: str = "LlamaDecoderLayer"
    quantized_model: str = "quantized_model"

    amp_dtype: Optional[str] = "float32"
    attn_implementation: Optional[str] = None
    code_dtype: Optional[str] = "uint16"
    limit_parallel_inits: int = 4
    load_dtype: str = "float32"
    master_dtype: str = "float32"
    model_seqlen: int = 4096
    monkeypatch_old_pickle: bool = False
    straight_through_buffer_dtype: Optional[str] = "float32"
    wrap_separately: Optional[Iterable] = None  # default: ()

    ### data
    dataset_name: Optional[Union[str, Path, Dataset]] = "dataset_name"

    cache_dir: Optional[Union[str, Path]] = None
    dataset_config_name: Optional[str] = None
    download_num_workers: Optional[int] = None
    eval_datasets: Optional[Iterable] = None  # default: ("wikitext2", "c4")
    num_workers: int = 8
    overwrite_cache: bool = False
    preprocessing_chunk_length: Optional[int] = None
    preprocessing_keep_in_memory: bool = False
    preprocessing_num_workers: Optional[int] = field(default_factory=os.cpu_count)
    save_dataset_and_exit: Optional[Union[str, Path]] = None
    split: str = "none"
    trust_remote_code: bool = True
    use_fast_tokenizer: bool = False
    skip_grouping: bool = False

    ### finetuning
    save: Optional[Union[str, Path]] = None

    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    amsgrad: bool = False
    batch_size: int = 256
    beam_size: int = 1
    code_adam_16bit: bool = False
    code_beta1: float = 0.0
    code_beta2: float = 0.95
    code_lr: float = 1e-3
    code_selection_temperature: int = 0
    code_trust_ratio: Optional[float] = 1e-2
    debias: Optional[bool] = True
    delta_decay: Optional[float] = 0
    eval_every_steps: Optional[int] = 1
    force_code_update: bool = False
    gradient_checkpointing: bool = True
    keep_best_model: bool = False
    lamb: bool = True
    lr: float = 1e-4
    max_code_change_per_step: float = 1e-2
    max_epochs: int = 10
    microbatch_size: Optional[int] = 4
    minimize_sync: bool = False
    on_save: Optional[str] = None
    print_every_steps: Optional[int] = 1
    save_every_steps: Optional[int] = 1
    seed: int = 1337
    update_codebooks_and_scales: bool = True
    update_codes: bool = True
    update_non_quantized_parameters: bool = True
    use_fsdp_amp: bool = False
    verbose_optimizer: bool = True
    wandb: bool = False

    def __post_init__(self):
        assert self.attn_implementation in [
            None,
            "eager",
            "flash_attention_2",
            "sdpa",
        ], f'Invalid attn_implementation: {self.attn_implementation}. Allowed values: [None, "eager", "flash_attention_2", "sdpa"]]'
        assert self.load_dtype in [
            "auto",
            "float16",
            "float32",
            "bfloat16",
        ], f'Invalid load_dtype: {self.load_dtype}. Allowed values: ["auto", "float16", "float32", "bfloat16"]'
        if not self.eval_datasets:
            self.eval_datasets = ["wikitext2", "c4"]
        if not self.wrap_separately:
            self.wrap_separately = []


@dataclass
class ConversionConfig:
    base_model: str = "base_model"
    quantized_model: str = "quantized_model"
    pv_fsdp_dir: Optional[Union[str, Path]] = None
    save: str = "save"

    attn_implementation: Optional[str] = None
    code_dtype: Optional[str] = "int32"
    load_dtype: str = "auto"
    monkeypatch_old_pickle: bool = False
    p_finetuned_state_dict: Optional[Union[str, Path]] = None
    trust_remote_code: bool = True

    def __post_init__(self):
        assert self.attn_implementation in [
            None,
            "eager",
            "flash_attention_2",
            "sdpa",
        ], f'Invalid attn_implementation: {self.attn_implementation}. Allowed values: [None, "eager", "flash_attention_2", "sdpa"]]'
        assert self.load_dtype in [
            "auto",
            "float16",
            "float32",
            "bfloat16",
        ], f'Invalid load_dtype: {self.load_dtype}. Allowed values: ["auto", "float16", "float32", "bfloat16"]'


@dataclass
class AQLMConfig:
    calibration_config: CalibrationConfig = field(default_factory=CalibrationConfig)
    finetune_config: FineTuneConfig = field(default_factory=FineTuneConfig)
    conversion_config: ConversionConfig = field(default_factory=ConversionConfig)
    save_intermediate_results: bool = False
    overwrite: bool = False

    def _overwrite_or_run(self, save_path: Union[str, Path]) -> bool:
        if not save_path:
            return self.overwrite
        save_path = Path(save_path)

        return (
            self.overwrite
            or not save_path.exists()
            or len(list(save_path.iterdir())) == 0
        )

    @cached_property
    def overwrite_or_run_caliberation(self):
        return self._overwrite_or_run(self.calibration_config.save)

    @cached_property
    def overwrite_or_run_conversion(self):
        return self._overwrite_or_run(self.conversion_config.save)

    @cached_property
    def overwrite_or_run_dataset_tokenization(self):
        return self._overwrite_or_run(self.finetune_config.dataset_name)

    @cached_property
    def overwrite_or_run_finetune(self):
        return self._overwrite_or_run(self.finetune_config.save)

    @cached_property
    def overwrite_or_run_all(self):
        return (
            self.overwrite_or_run_caliberation
            and self.overwrite_or_run_conversion
            and self.overwrite_or_run_dataset_tokenization
            and self.overwrite_or_run_finetune
        )
