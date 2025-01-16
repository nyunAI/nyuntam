from dataclasses import dataclass, field
from datasets import load_dataset, DatasetDict, load_from_disk
from pathlib import Path
from typing import List, Optional, Union
import gc
import torch

import logging

logger = logging.getLogger(__name__)


@dataclass
class Dataset:

    dataset_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Name of the dataset."}
    )
    dataset_subset: Optional[str] = field(
        default=None, metadata={"help": "Name of the dataset subset."}
    )
    text_column: Optional[str] = field(
        default="text",
        metadata={
            "help": "Name of the text column(s). If multiple columns, separate by comma."
        },
    )
    split: Optional[str] = field(
        default="train", metadata={"help": "Split of the dataset."}
    )
    format_string: Optional[str] = field(
        default=None, metadata={"help": "Format of the dataset."}
    )

    new_text_column = "text"

    @classmethod
    def set_new_text_column(cls, new_text_column: str):
        cls.new_text_column = new_text_column

    @staticmethod
    def text_columns(text_column: str) -> List[str]:
        "Return the text columns as a list."
        return text_column.split(",")

    @classmethod
    def from_name_or_path(
        cls,
        dataset_name: Optional[str],
        dataset_path: Optional[Path],
        dataset_subset: Optional[str],
        save_dir: Path,
        cache_dir: Path,
        split: str = "train",
        format_string: Optional[str] = None,
        text_column: str = "text",
    ):
        if (
            (dataset_path is None)
            or (not dataset_path.exists())
            or (len(list(dataset_path.iterdir())) == 0)
        ):
            path: Union[str, Path] = dataset_name
        else:
            path: Union[str, Path] = dataset_path

        if isinstance(path, str):
            try:
                ds = load_dataset(path=path, name=dataset_subset, cache_dir=cache_dir)
            except Exception as e:
                raise RuntimeError(
                    f"Error while loading dataset `{path}` with subset `{dataset_subset}` and split `{split}`"
                ) from e
        elif isinstance(path, Path):
            try:
                ds = load_from_disk(str(path))
            except Exception as e:
                raise RuntimeError(f"Error while loading dataset from {path}") from e

        text_columns = cls.text_columns(text_column)
        if len(text_columns) > 1:
            assert (
                format_string is not None
            ), "When multiple text columns are used, `format_string` is required."
        for ds_split in ds.keys():
            for k in text_columns:
                assert (
                    k in ds[ds_split][0].keys()
                ), f"The key '{k}' is not present in the dataset. The dataset keys(columns) are {ds[split][0].keys()}"

        cls._format_and_save(ds, text_columns, format_string, save_dir)
        del ds
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return cls(
            dataset_name_or_path=save_dir,
            dataset_subset=dataset_subset,
            text_column=cls.new_text_column,
            split=split,
            format_string=format_string,
        )

    @classmethod
    def _format_dataset(
        cls, ds: DatasetDict, text_columns: List[str], format_string: str
    ):
        if format_string is None:
            # since no format string is present, we will use the first text column as the new text column
            cls.set_new_text_column(text_columns[0])
            return ds

        try:
            for split in ds.keys():
                fs = format_string.format(**{k: ds[split][0][k] for k in text_columns})
        except KeyError as e:
            raise KeyError(
                f"The format_string expects a key that is not present in {text_columns}: {e}. The dataset keys(columns) are {ds[0].keys()}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Error while formatting the format_string with the dataset columns."
            ) from e

        logger.info(
            f"Found format string.\n"
            f"The dataset will be loaded using the format string formatted with {text_columns}.\n"
            f"Following is a sample of the formatted dataset:\n{fs}.\n"
        )

        new_text_column = cls.new_text_column
        for split in ds.keys():
            ds[split] = ds[split].map(
                lambda x: {
                    new_text_column: format_string.format(
                        **{k: x[k] for k in text_columns}
                    )
                }
            )
        return ds

    @classmethod
    def _format_and_save(
        cls,
        ds: DatasetDict,
        text_columns: List[str],
        format_string: str,
        save_dir: Path,
    ):
        ds = cls._format_dataset(ds, text_columns, format_string)
        cls._save_dataset(ds, save_dir)

    @staticmethod
    def _save_dataset(ds: DatasetDict, save_dir: Path):
        # skip if path is not empty; also list contents of the directory if not empty
        if save_dir.exists() and len(list(save_dir.iterdir())) > 0:
            logger.info(
                f"Path {save_dir} already exists and is not empty (Already cached)."
            )
            return
        try:
            ds.save_to_disk(str(save_dir))
        except Exception as e:
            raise RuntimeError(f"Error while saving the dataset to {save_dir}") from e

        logger.info(f"Dataset saved at {save_dir}.")

    def to_json_file(self, dir: Path):
        import json
        import os

        if not dir.exists():
            os.makedirs(dir, exist_ok=True)

        json_path = dir / "dataset.json"
        data = {
            "dataset_name_or_path": str(self.dataset_name_or_path.absolute()),
            "dataset_subset": self.dataset_subset,
            "text_column": self.text_column,
            "split": self.split,
            "format_string": self.format_string,
        }
        with open(json_path, "w") as file:
            json.dump(data, file, indent=4)

        return json_path

    # ===== custom loaders =====

    def load_caliberation_data(self):
        ds = load_from_disk(self.dataset_name_or_path)
        return ds[self.split][self.text_column]

    def get_flap_dataloader(self, nsamples=128, seed=0, seqlen=2048, tokenizer=None):

        import random

        ds = load_from_disk(self.dataset_name_or_path)
        traindata = ds[self.split]
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(
                    traindata[i][self.text_column], return_tensors="pt"
                )
                if trainenc.input_ids.shape[1] > seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

        # TODO: add logic to support testloader when eval is enabled (ref - FLAP.lib.data.get_loaders)
        return trainloader

    def load_split(self, split=None):
        if not split:
            split = self.split
        return load_from_disk(self.dataset_name_or_path)[split]

    def tokenize_aqlm(
        self, aqlm_config, tokenized_ds_path: Union[str, Path]
    ) -> Union[str, Path]:
        from copy import deepcopy
        from text_generation.quantization.aqlm import AQLMConfig, tokenize_dataset

        assert isinstance(
            aqlm_config, AQLMConfig
        ), "aqlm_config must be an instance of AQLMConfig"

        config = deepcopy(aqlm_config)
        config.finetune_config.load_dtype = "bfloat16"
        config.finetune_config.dataset_name = self
        config.finetune_config.save_dataset_and_exit = str(tokenized_ds_path)
        tokenize_dataset(config)
        del config
        return tokenized_ds_path

    def get_aqlm_caliberation_dataloader(
        self, nsamples=128, seqlen=2048, tokenizer=None
    ):
        import random
        from tqdm import trange

        assert (
            tokenizer is not None
        ), "Tokenizer must be provided for AQLM calibration dataloader"

        ds = load_from_disk(self.dataset_name_or_path)
        traindata = ds[self.split]
        not_none_else_val = lambda x, val: x if x is not None else val
        tokenizer.bos_token_id = not_none_else_val(tokenizer.bos_token_id, 1)
        tokenizer.eos_token_id = not_none_else_val(tokenizer.eos_token_id, 2)
        trainloader = []
        searched_indices = set()

        for _ in trange(
            nsamples,
            desc=f"Making {self.dataset_name_or_path} calibration set",
            leave=False,
        ):
            while True:
                if len(searched_indices) >= len(traindata):
                    logger.error(
                        f"Dataset {self.dataset_name_or_path} does not have enough samples with the required sequence length "
                        f"to create the calibration set. Total searched indices: {len(searched_indices)}, required nsamples: {nsamples}, "
                        f"seqlen: {seqlen}"
                    )
                    raise RuntimeError(
                        f"Dataset {self.dataset_name_or_path} does not have enough samples with the required sequence length "
                        f"to create the calibration set. Searched {len(searched_indices)} indices, "
                        f"needed {nsamples} samples with sequence length {seqlen}."
                    )

                i = random.randint(0, len(traindata) - 1)

                if i in searched_indices:
                    continue

                searched_indices.add(i)
                trainenc = tokenizer(
                    traindata[i][self.new_text_column], return_tensors="pt"
                )

                if trainenc.input_ids.shape[1] > seqlen:
                    break

            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]

            assert (
                inp.shape[1] == seqlen
            ), f"Extracted sequence length {inp.shape[1]} does not match the required {seqlen}"
            logger.debug(
                f"Added sequence from index {i} to {j} with shape {inp.shape} to calibration set."
            )
            trainloader.append(inp)

        logger.info(f"Calibration set created with {len(trainloader)} samples.")
        return trainloader
