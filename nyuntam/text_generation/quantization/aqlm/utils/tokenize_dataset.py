# Parts of this code are taken from https://github.com/nyunAI/AQLM/blob/pv-tuning/finetune_fsdp.py

from text_generation.core.dataset import Dataset
from text_generation.quantization.aqlm.config import AQLMConfig
from text_generation.quantization.aqlm.utils.distributed import get_rank

# quantization/aqlm/AQLM
from AQLM.src.datautils import (
    group_texts,
    split_long_texts,
)

import torch
import torch.distributed
import torch.utils
import torch.utils.data
import logging
from functools import partial
import transformers
from pathlib import Path

logger = logging.getLogger(__name__)


def tokenize_dataset(config: AQLMConfig):
    """Prepares dataset for AQLM PV-Tuning."""

    assert torch.cuda.is_available() and torch.distributed.is_available()
    rank = get_rank()
    if rank != 0:
        return
    world_size = torch.distributed.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    args = config.finetune_config

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
    assert tokenizer.eos_token_id is not None
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    assert (
        args.save_dataset_and_exit is not None
    ), "Please provide a path to save the dataset"
    save_path = Path(args.save_dataset_and_exit)
    if (
        not config.overwrite_or_run_all
        and not config.overwrite_or_run_dataset_tokenization
    ):
        logger.info(f"Skipping tokenization at save path: {save_path}")
        return

    assert isinstance(
        args.dataset_name, Dataset
    ), "dataset_name must be an instance of Dataset"
    dataset = args.dataset_name.load_split()

    def is_tokenized(dataset):
        return "input_ids" in dataset.column_names

    if is_tokenized(dataset):
        if rank == 0:
            dataset.save_to_disk(save_path)
            logger.info(f"Saved tokenized dataset to {save_path}")
        return

    text_column_name = args.dataset_name.new_text_column

    if args.preprocessing_chunk_length is not None:
        dataset = dataset.map(
            lambda examples: {
                text_column_name: split_long_texts(
                    examples[text_column_name], args.preprocessing_chunk_length
                )
            },
            batched=True,
            num_proc=(
                args.preprocessing_num_workers
                if args.preprocessing_num_workers is not None
                else args.num_workers
            ),
            keep_in_memory=args.preprocessing_keep_in_memory,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Splitting dataset over newline into chunks of ~{args.preprocessing_chunk_length} characters",
        )
    tokenized_dataset = dataset.map(
        lambda example: tokenizer(
            example[text_column_name],
            padding="max_length",
            max_length=args.model_seqlen,
            truncation=True,
        ),
        num_proc=(
            args.preprocessing_num_workers
            if args.preprocessing_num_workers is not None
            else args.num_workers
        ),
        remove_columns=list(dataset.column_names),
        keep_in_memory=args.preprocessing_keep_in_memory,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    if not args.skip_grouping:
        tokenized_dataset = tokenized_dataset.map(
            partial(group_texts, block_size=args.model_seqlen, add_labels=False),
            batched=True,
            num_proc=(
                args.preprocessing_num_workers
                if args.preprocessing_num_workers is not None
                else args.num_workers
            ),
            keep_in_memory=args.preprocessing_keep_in_memory,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {args.model_seqlen}",
        )
        assert is_tokenized(tokenized_dataset)

    if rank == 0:
        tokenized_dataset.save_to_disk(save_path)
        logger.info(f"Saved tokenized dataset to {save_path}")
    return tokenized_dataset
