# Some parts of this code are taken from https://github.com/Vahe1994/AQLM/blob/pv-tuning/main.py

from text_generation.quantization.aqlm.config import AQLMConfig, CalibrationConfig
from text_generation.quantization.aqlm.utils.distributed import get_rank

# quantization/aqlm/AQLM
from AQLM.main import get_model, quantize_aq, get_loaders, perplexity_eval
from AQLM.src.datautils import (
    get_loaders,
)
from AQLM.src.modelutils import get_model

import os
import torch
import logging
import transformers
from pathlib import Path

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

logger = logging.getLogger(__name__)


def caliberate_model(config: AQLMConfig):
    """AQLM calibration script for quantizing a model"""
    rank = get_rank()
    if rank != 0:
        return

    torch.set_num_threads(min(16, torch.get_num_threads()))
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    args: CalibrationConfig = config.calibration_config

    if not config.overwrite_or_run_all and not config.overwrite_or_run_caliberation:
        logger.info("Skipping caliberation")
        return

    if args.devices is None:
        if torch.cuda.is_available():
            args.devices = [
                torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
            ]
        else:
            args.devices = [torch.device("cpu")]
    else:
        args.devices = [torch.device(device_str) for device_str in args.devices]
    assert all(isinstance(device, torch.device) for device in args.devices)

    if args.wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        args.exp_name = (
            os.environ.get("WANDB_NAME", "AQ")
            + f"_num_codebooks_{args.num_codebooks}"
            + f"_out_group_size_{args.out_group_size}"
            + f"_in_group_size_{args.in_group_size}"
            + f"_nbits_per_codebook_{args.nbits_per_codebook}"
            + f"_codebook_value_nbits_{args.codebook_value_nbits}"
            + f"_codebook_value_num_groups_{args.codebook_value_num_groups}"
            + f"_scale_nbits_{args.scale_nbits}"
            + f"_steps_per_epoch_{args.steps_per_epoch}"
            + f"_init_max_iter{args.init_max_iter}"
            + f"_{len(args.devices)}gpus"
        )
        wandb.init(
            config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")},
        )

    model = get_model(
        args.model_path,
        args.load,
        args.dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    ).train(False)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)
    if not args.load and not args.no_quant:
        logger.info("Loading data...")
        data = args.job_dataset.get_aqlm_caliberation_dataloader(
            nsamples=args.nsamples, seqlen=args.model_seqlen, tokenizer=tokenizer
        )
        del tokenizer
        if args.val_size > 0:
            all_ids = torch.randperm(len(data))
            train_ids, val_ids = all_ids[args.val_size :], all_ids[: args.val_size]
            train_data = [data[i] for i in train_ids]
            val_data = [data[i] for i in val_ids]
        else:
            train_data = data
            val_data = None
        logger.info("Quantizing model...")
        results = quantize_aq(model, train_data, val_data, args)

    logger.info("Evaluating perplexity...")
    torch.cuda.reset_peak_memory_stats()
    datasets = ["wikitext2", "c4"]
    if args.new_eval:
        datasets = ["wikitext2", "c4-new"]
    for dataset in datasets:
        testloader = get_loaders(
            dataset,
            seed=args.seed,
            model_path=args.model_path,
            seqlen=args.model_seqlen,
            eval_mode=True,
            use_fast_tokenizer=args.use_fast_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
        args.dataset_name = dataset
        logger.info(f"\nEvaluating perplexity for {args.dataset_name} dataset ...")
        ppl = perplexity_eval(model, testloader, args)
        logger.info(f"\n{args.dataset_name} perplexity = {ppl:.4f}\n")

    logger.info(f"eval: {torch.cuda.max_memory_allocated()=:,}")
    if args.wandb:
        wandb.log(
            {"max_cuda_mem_eval": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        )
