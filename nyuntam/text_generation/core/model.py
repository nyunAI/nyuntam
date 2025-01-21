from pathlib import Path
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import OrderedDict
import torch
import gc
from typing import Optional
from math import floor

import logging

logger = logging.getLogger(__name__)


ALLOWED_EXTENSIONS = {".pt", ".bin"}


# =========== Exceptions ===========
class CustomModelLoadError(RuntimeError):
    """Exception for custom model loading errors."""

    pass


class NamedModelLoadError(RuntimeError):
    """Exception for named model loading errors."""

    pass


# =========== Language Model ===========
@dataclass
class LanguageModel:
    """Language model class for Kompress."""

    model_name: str = field(default=None, metadata={"help": "Name of the model"})
    model_path: Path = field(
        default=None,
        metadata={
            "help": "Path to the model directory saved with `.save_pretrained()`."
        },
    )
    size: str = field(default=None, metadata={"help": "Size of the model."})

    @staticmethod
    def get_model_size(model: torch.nn.Module):
        BILLION = 1_000_000_000
        return floor(sum(p.numel() for p in model.parameters()) / BILLION)

    @classmethod
    def from_model_name(
        cls,
        model_name: str,
        dtype: torch.dtype,
        save_dir: Path,
        cache_dir: Path,
        patch_phi3: bool = False,
    ):
        """To load a Huggingface language model for Kompress. \
            Loads a model from the Huggingface model hub and saves it in the cache directory for consistent loading accross downstream tasks.

        Args:
            `model_name (*required)`: Name of the model
            `save_dir (*required)`: Path to save the model. (stores with `.save_pretrained()`)
        """
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, cache_dir=cache_dir
            )
            patch_phi3 = patch_phi3 and model.config.model_type == "phi3"
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        except Exception as e:
            raise RuntimeError(f"Error while loading model {model_name}") from e

        if patch_phi3:
            cls._patch_phi3_to_llama_and_save(model, tokenizer, dtype, save_dir)
        else:
            cls._save_model(model, dtype, tokenizer, save_dir)

        size_key = f"{cls.get_model_size(model)}B"
        if size_key.lower() not in model_name.lower():
            model_name = f"{model_name}-{size_key}"

        # clean model and tokenizer
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return cls(model_name=model_name, model_path=save_dir, size=size_key)

    @classmethod
    def from_model_path(
        cls,
        model_name: str,
        custom_model_path: Path,
        dtype: torch.dtype,
        save_dir: Path,
        cache_dir: Path,
        patch_phi3: bool = False,
    ):
        """To load a custom language model for Kompress. \
            Loads a model from a given path and saves it in the cache directory for consistent loading accross downstream tasks.

        Args:
            `model_name (*required)`: Name of the model
            `custom_model_path (*required)`: Path to 
                    a model directory saved with `.save_pretrained()`, or \
                    a complete HF loadable model saved with `torch.save()`, or \
                    path to `state_dict` of a model \
                    (accepts `.pt` and `.bin` extensions).
            `save_dir (*required)`: Path to save the model. (stores with `.save_pretrained()`)
        """
        if not custom_model_path.exists():
            raise FileNotFoundError(f"Model path {custom_model_path} does not exist.")

        # custom model path is always a directory. Check if config.json exists otherwise change model path to .pt or .bin file inside the directory
        for file in custom_model_path.iterdir():
            if file.name == "config.json":
                custom_model_path = file.parent
                break
            elif file.suffix in ALLOWED_EXTENSIONS:
                custom_model_path = file
                break
        if custom_model_path.is_dir():
            # Load model from a directory saved with `.save_pretrained()`
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    custom_model_path, cache_dir=cache_dir
                )
                patch_phi3 = patch_phi3 and model.config.model_type == "phi3"
            except Exception as e:
                raise CustomModelLoadError(
                    f"Error while loading custom model from {custom_model_path}"
                ) from e

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    custom_model_path, cache_dir=cache_dir
                )
            except Exception as e:
                logging.warning(
                    f"No tokenizer found at {custom_model_path}. Loading instead for {model_name}."
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, cache_dir=cache_dir
                )

        elif custom_model_path.is_file():

            if not custom_model_path.suffix in ALLOWED_EXTENSIONS:
                raise ValueError(
                    f"Invalid file extension for model file {custom_model_path}. Allowed extensions: {ALLOWED_EXTENSIONS}"
                )

            try:
                load = torch.load(custom_model_path)
                if isinstance(load, OrderedDict):  # load is a state_dict
                    logger.info(f"Loading model from state_dict at {custom_model_path}")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name, state_dict=load, cache_dir=cache_dir
                    )
                    patch_phi3 = patch_phi3 and model.config.model_type == "phi3"
                elif isinstance(load, torch.nn.Module):  # load is a complete model
                    logger.info(f"Loading model at {custom_model_path}")
                    model = load
                    patch_phi3 = patch_phi3 and "phi3" in model_name.lower()

                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, cache_dir=cache_dir
                )
            except Exception as e:
                raise NamedModelLoadError(
                    f"Error while loading named model from {custom_model_path}"
                ) from e

        if patch_phi3:
            cls._patch_phi3_to_llama_and_save(model, tokenizer, dtype, save_dir)
        else:
            cls._save_model(model, dtype, tokenizer, save_dir)

        size_key = f"{cls.get_model_size(model)}B"
        if size_key.lower() not in model_name.lower():
            model_name = f"{model_name}-{size_key}"

        # clean model and tokenizer
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return cls(model_name=model_name, model_path=save_dir, size=size_key)

    @classmethod
    def from_model_path_or_name(
        cls,
        model_name: str,
        custom_model_path: Optional[Path],
        dtype: torch.dtype,
        save_dir: Path,
        cache_dir: Path,
        patch_phi3: bool = False,
    ):
        """To load a custom language model for Kompress. \
            Loads a model from a given path and saves it in the cache directory for consistent loading accross downstream tasks.

        Args:
            `model_name (*required)`: Name of the model
            `custom_model_path (*optional*)`: Path to 
                    a model directory saved with `.save_pretrained()`, or \
                    a complete HF loadable model saved with `torch.save()`, or \
                    path to `state_dict` of a model \
                    (accepts `.pt` and `.bin` extensions).
            `save_dir (*required)`: Path to save the model. (stores with `.save_pretrained()`)
        """
        if (
            (custom_model_path is None)
            or (not custom_model_path.exists())
            or (len(list(custom_model_path.iterdir())) == 0)
        ):
            # if
            return cls.from_model_name(
                model_name, dtype, save_dir, cache_dir, patch_phi3=patch_phi3
            )
        else:
            return cls.from_model_path(
                model_name,
                custom_model_path,
                dtype,
                save_dir,
                cache_dir,
                patch_phi3=patch_phi3,
            )

    @staticmethod
    def _save_model(model, dtype, tokenizer, save_dir: Path):
        """Save model and tokenizer to the given directory."""

        # skip if path is not empty; also list contents of the directory if not empty
        if save_dir.exists() and len(list(save_dir.iterdir())) > 0:
            logger.info(
                f"Path {save_dir} already exists and is not empty (Already cached)."
            )
            return

        if dtype:
            model.to(dtype)
        try:
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            torch.save(tokenizer, save_dir / "tokenizer.model")  # for lmquant, qserve
            logger.info(f"Model saved at {save_dir}.")
        except Exception as e:
            raise RuntimeError(f"Error while caching model to {save_dir}.") from e

    @staticmethod
    def _patch_phi3_to_llama_and_save(model, tokenizer, dtype, save_dir: Path):
        """Patch Phi3 model to Llama model and save it to the given directory."""
        # TODO add the patch_modules function to the main codebase
        raise NotImplementedError("Patch modules function not implemented yet.")

        import json

        patch_modules(model, PHI3_PATCH_TYPES)
        phi3_config = model.config.to_dict()
        _ = phi3_config.pop("auto_map")
        _ = phi3_config.pop("rope_scaling")
        phi3_config["architectures"] = ["LlamaForCausalLM"]
        phi3_config["model_type"] = "llama"
        phi3_config["_name_or_path"] = str(save_dir)

        LanguageModel._save_model(model, dtype, tokenizer, save_dir)

        (save_dir / "config.json").unlink()
        with open(save_dir / "config.json", "w") as f:
            json.dump(phi3_config, f, indent=4)
        logger.info(f"Saved patched phi3 model to: {save_dir}")

        del model
        del phi3_config
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
