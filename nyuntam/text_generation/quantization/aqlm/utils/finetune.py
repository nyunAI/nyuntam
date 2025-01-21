# Parts of this code are taken from https://github.com/nyunAI/AQLM/blob/pv-tuning/finetune_fsdp.py

from text_generation.quantization.aqlm.config import AQLMConfig
from text_generation.quantization.aqlm.utils.distributed import get_rank

# quantization/aqlm/AQLM
from AQLM.finetune_fsdp import (
    _load_state,
    _save_state,
    load_base_model,
    load_dequantized_model,
    compute_loss_on_batch,
    compute_validation_perplexities,
)
from AQLM.src.aq_ops import master_rank_first, one_rank_at_a_time
from AQLM.src.datautils import (
    get_wikitext2,
    get_red_pajama,
    get_ptb,
    get_ptb_new,
    get_c4,
    get_c4_new,
    set_seed,
)
from AQLM.src.pv_utils import (
    get_original_named_parameters_from_fsdp_module,
)
from AQLM.src.pv_optimizer import StraightThroughAdamW

# imports with src.aq path to avoid assertion errors in AQLM/
from src.aq import QuantizedWeight
from src.pv_utils import (
    split_quantized_weights_between_ranks,
    YourQuantizedWeightIsInAnotherRank,
)

import torch
import torch.distributed
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    StateDictType,
    FullStateDictConfig,
)
import torch.utils
import torch.utils.data
import logging
import transformers
from tqdm import tqdm
from contextlib import nullcontext
from datasets import load_from_disk

from lm_eval.models.huggingface import HFLM
from lm_eval.evaluator import simple_evaluate
import sys
import gc
import os

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

logger = logging.getLogger(__name__)


METRIC_EVAL_DATASETS = [
    "arc_challenge",
    "winogrande",
    "gsm8k",
]


# overwrite get_loaders from AQLM to add support for metric evaluation datasets
def get_loaders(
    name,
    nsamples=128,
    seed=0,
    seqlen=2048,
    eval_mode=False,
    model_path=None,
    use_fast_tokenizer=False,
    trust_remote_code=None,
):
    set_seed(seed)

    if name.lower() == "none":
        print(
            "Not loading any dataset. (OK if you use no compression or methods like RTN.)"
        )
        return None
    elif os.path.isfile(name):
        try:
            data = torch.load(name)[:nsamples]
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Failed to load custom data from {name}.",
                "Check data path or use one of [c4, wikitext2, ptb, pajama, none]",
            )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path, use_fast=use_fast_tokenizer, trust_remote_code=trust_remote_code
        )

        if name.lower() == "wikitext2":
            data = get_wikitext2(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "pajama":
            data = get_red_pajama(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "ptb":
            data = get_ptb(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "ptb_new":
            data = get_ptb_new(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "c4":
            data = get_c4(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "c4_new":
            data = get_c4_new(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif any(metric in name.lower() for metric in METRIC_EVAL_DATASETS):
            print(
                f"{name} is a metric evaluation dataset. Will be loaded in the eval loop."
            )
            return None
        else:
            raise ValueError(
                f"Failed to load data from {name}.",
                "Check dataset name or path or use one of [c4, wikitext2, ptb, pajama, none]",
            )

    if hasattr(data, "input_ids"):
        data = data.input_ids

    print(f"Loaded data from {name}; {len(data)=} sequences")
    return data


# overwrite _save_model from AQLM to save model state dict
def _save_model(
    args,
    dequantized_model: FullyShardedDataParallel,
    optimizer: StraightThroughAdamW,
    step,
):
    """Save consolidated model state dict"""
    output_path = os.path.join(args.save, f"best_model_{step}")
    os.makedirs(output_path, exist_ok=True)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    local_quantized_weight_names = set()
    for name, quantized_weight in optimizer.iterate_local_quantized_weights():
        torch.save(quantized_weight, os.path.join(output_path, f"{name}.pth"))
        local_quantized_weight_names.add(name)

    quantized_weight_names_by_rank = (
        [None for _ in range(world_size)] if rank == 0 else None
    )
    torch.distributed.gather_object(
        local_quantized_weight_names, quantized_weight_names_by_rank, dst=0
    )

    with FullyShardedDataParallel.state_dict_type(
        dequantized_model,
        StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        model_state_dict = dequantized_model.state_dict()
        if rank == 0:
            all_quantized_weight_names = set()
            for local_quantized_weight_names in quantized_weight_names_by_rank:
                all_quantized_weight_names |= set(local_quantized_weight_names)

            non_quantized_state_dict = dict()
            for name, tensor in model_state_dict.items():
                if name in all_quantized_weight_names:
                    all_quantized_weight_names.remove(
                        name
                    )  # do not save de-quantized versions of quantized weights
                else:
                    non_quantized_state_dict[name] = tensor
            assert (
                len(all_quantized_weight_names) == 0
            ), f"mismatched names: {all_quantized_weight_names}"
            torch.save(
                non_quantized_state_dict,
                os.path.join(output_path, "non_quantized_state_dict.pth"),
            )
            print(f"Saved best model to {output_path}")


@torch.inference_mode()
def compute_evaluation_metrics(
    args,
    model: torch.nn.Module,
    eval_datasets: dict,
    tokenizer: transformers.PreTrainedTokenizer,
):
    rank = get_rank()
    metrics = {}
    for dataset_name, eval_dataset in eval_datasets.items():
        if rank == 0:
            print(f"Evaluating metrics on {dataset_name} ...")
        device = next(model.parameters()).device

        task_name, n_shot = dataset_name.split(":")
        n_shot = int(n_shot)

        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            max_batch_size=256,
            batch_size=2,
        )

        results = simple_evaluate(
            model=lm,
            tasks=[task_name],
            num_fewshot=n_shot,
            batch_size=2,
            device=device,
        )
        metrics[dataset_name] = results["results"][task_name].get(
            "acc_norm,none",
            results["results"][task_name]["acc,none"],
        )  # acc_norm,none is the normalized accuracy (not always present); acc,none is the raw accuracy (always present)
        if rank == 0:
            logger.info(f"{dataset_name} accuracy: {metrics[dataset_name]:.9f}")
        del lm, results, model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
    return metrics


def finetune_quantized(config: AQLMConfig):
    """Finetunes an AQLM quantized model with PV-Tuning."""

    assert torch.cuda.is_available() and torch.distributed.is_available()

    rank = get_rank()
    world_size = torch.distributed.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    args = config.finetune_config

    if not config.overwrite_or_run_all and not config.overwrite_or_run_finetune:
        logger.info("Skipping finetuning")
        return

    assert args.batch_size is not None, "please specify batch size"
    assert args.batch_size % world_size == 0
    if args.microbatch_size is None:
        args.microbatch_size = args.batch_size // world_size
    assert args.batch_size % (world_size * args.microbatch_size) == 0
    grad_accumulation_steps = args.batch_size // (world_size * args.microbatch_size)

    args.load_dtype = (
        getattr(torch, args.load_dtype) if args.load_dtype != "auto" else "auto"
    )
    args.amp_dtype = (
        getattr(torch, args.amp_dtype) if args.amp_dtype is not None else None
    )
    args.code_dtype = (
        getattr(torch, args.code_dtype) if args.code_dtype is not None else None
    )
    args.master_dtype = getattr(torch, args.master_dtype)
    if args.straight_through_buffer_dtype is not None:
        args.straight_through_buffer_dtype = getattr(
            torch, args.straight_through_buffer_dtype
        )
    else:
        args.straight_through_buffer_dtype = args.master_dtype

    if args.save_every_steps is not None:
        assert (
            args.save is not None
        ), f"save_every_steps={args.save_every_steps}, but save path not specified"
    if args.keep_best_model:
        assert args.save is not None, f"keep_best_model requires save path"
        assert (
            args.eval_every_steps is not None
        ), f"keep_best_model requires eval_every_steps"
        assert args.eval_datasets is not None, f"keep_best_model requires eval_datasets"

    if args.wandb and rank == 0:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        wandb.init(
            config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")}
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
    assert tokenizer.eos_token_id is not None
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with master_rank_first(local=True):
        dataset = load_from_disk(args.dataset_name)

    sampler = torch.utils.data.DistributedSampler(
        dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.seed
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.microbatch_size,
        num_workers=args.num_workers,
        sampler=sampler,
        collate_fn=transformers.default_data_collator,
    )
    eval_datasets = {
        dataset_name: get_loaders(
            dataset_name,
            seed=args.seed,
            model_path=args.base_model,
            seqlen=args.model_seqlen,
            eval_mode=True,
        )
        for dataset_name in args.eval_datasets
    }

    with one_rank_at_a_time(local=True, group_size=args.limit_parallel_inits):
        base_model = load_base_model(args, device)
        dequantized_model, named_quantized_params = load_dequantized_model(args, device)
        named_dequantized_params = get_original_named_parameters_from_fsdp_module(
            dequantized_model
        )
        assert all(name in named_dequantized_params for name in named_quantized_params)

        if world_size > 1:
            # distributed pv: each rank holds a subset of all quantized weights; the rest are replaced with pointers
            named_quantized_params = split_quantized_weights_between_ranks(
                named_quantized_params, verify_checksums=True
            )
        for quantized_weight in named_quantized_params.values():
            if isinstance(quantized_weight, QuantizedWeight):
                quantized_weight.to(device)
            else:
                assert isinstance(
                    quantized_weight, YourQuantizedWeightIsInAnotherRank
                ), f"rank {rank}. type: {type(quantized_weight)}"

    optimizer = StraightThroughAdamW(
        named_dequantized_params=named_dequantized_params,
        named_quantized_params=named_quantized_params,
        update_codes=(
            dict(
                lr=args.code_lr,
                betas=(args.code_beta1, args.code_beta2),
                lamb=args.lamb,
                debias=args.debias,
                amsgrad=args.amsgrad,
                compute_dtype=args.master_dtype,
                exp_avg_dtype=(
                    torch.float16 if args.code_adam_16bit else args.master_dtype
                ),
                exp_avg_sq_dtype=(
                    torch.bfloat16 if args.code_adam_16bit else args.master_dtype
                ),
                v_hat_max_dtype=(
                    torch.float16 if args.code_adam_16bit else args.master_dtype
                ),
            )
            if args.update_codes
            else None
        ),
        update_codebooks_and_scales=(
            dict(
                lr=args.lr,
                betas=(args.adam_beta1, args.adam_beta2),
                lamb=args.lamb,
                debias=args.debias,
                amsgrad=args.amsgrad,
                compute_dtype=args.master_dtype,
                exp_avg_dtype=args.master_dtype,
                exp_avg_sq_dtype=args.master_dtype,
                v_hat_max_dtype=args.master_dtype,
            )
            if args.update_codebooks_and_scales
            else None
        ),
        update_non_quantized_parameters=(
            dict(
                lr=args.lr,
                betas=(args.adam_beta1, args.adam_beta2),
                lamb=args.lamb,
                debias=args.debias,
                amsgrad=args.amsgrad,
                compute_dtype=args.master_dtype,
                exp_avg_dtype=args.master_dtype,
                exp_avg_sq_dtype=args.master_dtype,
                v_hat_max_dtype=args.master_dtype,
            )
            if args.update_non_quantized_parameters
            else None
        ),
        delta_decay=args.delta_decay,
        max_code_change_per_step=args.max_code_change_per_step,
        force_code_update=args.force_code_update,
        code_trust_ratio=args.code_trust_ratio,
        beam_size=args.beam_size,
        straight_through_buffer_dtype=args.straight_through_buffer_dtype,
        verbose=args.verbose_optimizer,
    )
    del named_quantized_params

    metadata = dict(
        current_epoch=0,
        microbatches_since_epoch_start=0,
        total_microbatches=0,
        total_optimizer_steps=0,
        loss_numerator=0,
        loss_denominator=0,
        aggregated_loss=float("nan"),
        grad_steps_accumulated=0,
        early_stop_on=next(iter(args.eval_datasets)) if args.eval_datasets else None,
        best_eval_perplexity=float("inf"),
        best_eval_metric=sys.float_info.min,  ## sys.float_info.min is the smallest positive float; to be used for non perplexity metrics
        best_step=0,
    )

    _load_state(args, metadata, dequantized_model, optimizer)
    torch.distributed.barrier()

    for current_epoch in range(args.max_epochs):
        if current_epoch < metadata["current_epoch"]:
            continue  # skip finished epochs
        sampler.set_epoch(current_epoch)

        batch_iter = (
            tqdm(train_dataloader, desc=f"Training epoch #{current_epoch}")
            if rank == 0
            else train_dataloader
        )
        for batch_index, batch in enumerate(batch_iter):
            if batch_index <= metadata["microbatches_since_epoch_start"]:
                continue  # skip batches processed before checkpoint
            metadata["microbatches_since_epoch_start"] += 1
            metadata["total_microbatches"] += 1

            batch = {k: v.to(device) for k, v in batch.items()}
            loss = compute_loss_on_batch(
                batch, base_model, dequantized_model, amp_dtype=args.amp_dtype
            )  ## add support for DPO

            metadata["loss_numerator"] += loss.item()
            metadata["loss_denominator"] += 1
            metadata["grad_steps_accumulated"] += 1
            if metadata["grad_steps_accumulated"] < grad_accumulation_steps:
                with (
                    dequantized_model.no_sync() if args.minimize_sync else nullcontext()
                ):
                    (loss / grad_accumulation_steps).backward()
            else:
                (loss / grad_accumulation_steps).backward()
                optimizer.step()
                optimizer.zero_grad()
                metadata["grad_steps_accumulated"] = 0
                metadata["total_optimizer_steps"] += 1

                if (
                    args.print_every_steps
                    and metadata["total_optimizer_steps"] % args.print_every_steps == 0
                ):
                    loss_numerator_and_denominator = torch.tensor(
                        [metadata["loss_numerator"], metadata["loss_denominator"]],
                        dtype=torch.float64,
                        device=device,
                    )

                    torch.distributed.all_reduce(
                        loss_numerator_and_denominator,
                        op=torch.distributed.ReduceOp.SUM,
                    )
                    loss_numerator, loss_denominator = (
                        loss_numerator_and_denominator.tolist()
                    )
                    metadata["aggregated_loss"] = loss_numerator / loss_denominator
                    metadata["loss_numerator"] = metadata["loss_denominator"] = 0
                    if rank == 0:
                        logger.info(
                            f"epoch {metadata['current_epoch']}\tbatch {batch_index}"
                            f"\t| total updates = {metadata['total_optimizer_steps']}"
                            f"\tloss = {metadata['aggregated_loss']:.9f}"
                        )

                if (
                    args.eval_every_steps
                    and metadata["total_optimizer_steps"] % args.eval_every_steps == 0
                ):
                    if metadata["early_stop_on"] in ["wikitext2", "c4"]:
                        perplexity_scores = compute_validation_perplexities(
                            args, dequantized_model, eval_datasets
                        )
                        for dataset_name, perplexity in perplexity_scores.items():
                            metadata[f"perplexity_{dataset_name}"] = perplexity
                        metric_name = metadata["early_stop_on"]
                        if (
                            perplexity_scores[metric_name]
                            < metadata["best_eval_perplexity"]
                        ):
                            if rank == 0:
                                logger.info(
                                    f"New best perplexity ({metric_name}) = {perplexity_scores[metric_name]:.9f}"
                                )
                            metadata["best_eval_perplexity"] = perplexity_scores[
                                args.eval_datasets[0]
                            ]
                            metadata["best_step"] = metadata["total_optimizer_steps"]
                            if args.keep_best_model:
                                _save_model(
                                    args,
                                    dequantized_model,
                                    optimizer,
                                    metadata["total_optimizer_steps"],
                                )
                    else:
                        # early stop on other metrics (arc_challenge, winogrande, gsm8k)
                        eval_metrics = compute_evaluation_metrics(
                            args, dequantized_model, eval_datasets, tokenizer=tokenizer
                        )
                        for dataset_name, metric in eval_metrics.items():
                            metadata[f"metric_{dataset_name}"] = metric
                        metric_name = metadata["early_stop_on"]
                        if eval_metrics[metric_name] > metadata["best_eval_metric"]:
                            if rank == 0:
                                logger.info(
                                    f"New best metric ({metric_name}) = {eval_metrics[metric_name]:.9f}"
                                )
                            metadata["best_eval_metric"] = eval_metrics[metric_name]
                            metadata["best_step"] = metadata["total_optimizer_steps"]
                            if args.keep_best_model:
                                _save_model(
                                    args,
                                    dequantized_model,
                                    optimizer,
                                    metadata["total_optimizer_steps"],
                                )

                if args.wandb and rank == 0:
                    wandb.log(metadata, step=metadata["total_microbatches"])
                if (
                    args.save_every_steps
                    and metadata["total_optimizer_steps"] % args.save_every_steps == 0
                ):
                    _save_model(
                        args,
                        dequantized_model,
                        optimizer,
                        metadata["total_optimizer_steps"],
                    )
                    _save_state(args, metadata, dequantized_model, optimizer)

        metadata["microbatches_since_epoch_start"] = 0
        metadata["current_epoch"] += 1

    _save_state(args, metadata, dequantized_model, optimizer)
    logger.info("Finished training")
