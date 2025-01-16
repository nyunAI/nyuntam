# Parts of this code are taken from https://github.com/nyunAI/AQLM/blob/pv-tuning/convert_legacy_model_format.py

from text_generation.quantization.aqlm.config import AQLMConfig
from text_generation.quantization.aqlm.utils.distributed import get_rank

# quantization/aqlm/AQLM
from AQLM.convert_legacy_model_format import (
    load_quantized_model_from_fdsp_checkpoint,
    save_quantized_model,
)
from AQLM.src.aq import QuantizedWeight
from AQLM.src.aq_ops import is_signed

import torch
import torch.distributed
import torch.utils
import torch.utils.data
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


def convert_to_hf(config: AQLMConfig):
    """Converts an FSDP finetuned quantized model to Hugging Face format."""
    rank = get_rank()
    if rank != 0:
        return
    args = config.conversion_config
    if not config.overwrite_or_run_all and not config.overwrite_or_run_conversion:
        logger.info("Skipping conversion")
        return

    assert args.pv_fsdp_dir, "FSDP checkpoint directory must be specified"

    args.load_dtype = (
        getattr(torch, args.load_dtype) if args.load_dtype != "auto" else "auto"
    )
    args.code_dtype = (
        getattr(torch, args.code_dtype) if args.code_dtype is not None else None
    )

    quantized_model = load_quantized_model_from_fdsp_checkpoint(
        args.base_model,
        args.pv_fsdp_dir,
        dtype=args.load_dtype,
        trust_remote_code=args.trust_remote_code,
    )

    for module in quantized_model.modules():
        if isinstance(module, QuantizedWeight):
            if not hasattr(module, "codes_storage"):
                module.codes_storage = None
            if module.codes is None:
                module.unwrap_codes_()
            assert module.codes is not None
            if args.code_dtype is not None:
                assert module.nbits_per_codebook <= torch.iinfo(
                    args.code_dtype
                ).bits - is_signed(args.code_dtype)
                module.codes = nn.Parameter(
                    module.codes.to(args.code_dtype),
                    requires_grad=module.codes.requires_grad,
                )

    save_quantized_model(quantized_model, args.save)
    logger.info(f"Saved quantized model to {args.save}")
