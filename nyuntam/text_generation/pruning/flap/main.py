from text_generation.core.job import LMJob
from text_generation.core.dataset import Dataset
from text_generation.utils import create_instance, log_dict
from text_generation.pruning.flap.config import FlapConfig

# nyuntam
from nyuntam.algorithm import TextGenerationAlgorithm
from nyuntam.constants.keys import AdaptTasks

# pruning/flap/FLAP
from FLAP.lib.layerwrapper import BiasGPT
from FLAP.lib.prune import (
    metrics,
    find_layers,
    prepare_calibration_input,
    compress,
)

import os
import gc
import torch
import numpy as np
import json
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import asdict
from typing import Union, Optional

import logging

logger = logging.getLogger(__name__)

# wandb offline
os.environ["WANDB_MODE"] = "offline"


def free(times=2):
    for _ in range(times):
        torch.cuda.empty_cache()
        gc.collect()


class Pruner:
    def __init__(self, args: FlapConfig, job: LMJob, **kwargs):
        self.args = args
        self.job = job
        self.kw = kwargs
        self.output_dir = self.job.user_dir.output

        self.device_to_use = self.job.environment.cuda_device_ids[0]

    def prune(self):
        # method adapted from FLAP.main
        np.random.seed(self.args.seed)
        torch.random.manual_seed(self.args.seed)
        dtype = eval(f'torch.{self.kw.get("dtype", "float16")}')
        logger.info("Loading model..")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.job.model.model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True,
        )
        device = torch.device(f"cuda:{self.device_to_use}")

        for i in range(len(self.model.model.layers)):
            # set the bias to zero for self_attn.o_proj and mlp.down_proj
            # for a detailed explanation, refer - https://github.com/nyunAI/nyuntam-text-generation/pull/5#discussion_r1705901386
            self.model.model.layers[i].self_attn.o_proj.bias = torch.nn.Parameter(
                torch.zeros(
                    self.model.model.layers[i].self_attn.o_proj.weight.shape[0],
                    device=self.model.model.layers[i].self_attn.o_proj.weight.device,
                    dtype=dtype,
                )
            )
            self.model.model.layers[i].mlp.down_proj.bias = torch.nn.Parameter(
                torch.zeros(
                    self.model.model.layers[i].mlp.down_proj.weight.shape[0],
                    device=self.model.model.layers[i].mlp.down_proj.weight.device,
                    dtype=dtype,
                )
            )
            torch.nn.init.zeros_(self.model.model.layers[i].self_attn.o_proj.bias)
            torch.nn.init.zeros_(self.model.model.layers[i].mlp.down_proj.bias)

        self.model.seqlen = 128
        self.model.eval()

        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.job.model.model_path)

        device = self.model.hf_device_map.get("lm_head", self.model.device)
        logger.info("Pruning model...")
        self.prune_flap(
            args=self.args,
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=self.job.dataset,
            device=device,
        )

        logger.info("*" * 30)
        logger.info(
            f"pruned model parameter {sum(p.numel() for p in self.model.parameters()) / 1000 ** 3:.2f}B"
        )
        logger.info("*" * 30)

        try:
            torch.save(self.model, self.output_dir / "wds.pt")
        except Exception as e:
            # TODO: fix torch.save when pruning on multiple GPUs
            if (self.output_dir / "wds.pt").exists():
                (self.output_dir / "wds.pt").unlink()
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            logger.warn(
                f"couldn't save with torch.save, saved with save_pretrained instead"
            )
        for attr in ["_config", "_config_path"]:
            if hasattr(self.args, attr):
                delattr(self.args, attr)
        with open(self.output_dir / "prune_config.json", "w") as f:
            json.dump(asdict(self.args), f, indent=4)
        logger.info("Model pruned")

        del self.model
        del self.tokenizer

    def prune_flap(
        self,
        args: FlapConfig,
        model: Union[torch.nn.Module, AutoModelForCausalLM],
        tokenizer: AutoTokenizer,
        device=torch.device("cuda:0"),
        dataset: Optional[Dataset] = None,
    ):
        # method adapted from FLAP.lib.prune
        use_cache = model.config.use_cache
        model.config.use_cache = False
        dataloader = dataset.get_flap_dataloader(
            nsamples=args.nsamples,
            seed=args.seed,
            seqlen=model.seqlen,
            tokenizer=tokenizer,
        )

        with torch.no_grad():
            inps, outs, position_ids = prepare_calibration_input(
                model, dataloader, device
            )
        layers = model.model.layers

        attn_metric_list, mlp_metric_list = [], []
        attn_baseline_inp_list, mlp_baseline_inp_list = [], []
        attn_mask, mlp_mask = [], []

        # Split into sub-problems, separate statistics for each module
        for i in trange(
            args.start_pruning_layer_idx, len(layers), desc="Processing layers"
        ):
            layer = layers[i]
            subset = {}
            subset.update({"self_attn.o_proj": find_layers(layer)["self_attn.o_proj"]})
            subset.update({"mlp.down_proj": find_layers(layer)["mlp.down_proj"]})

            if f"model.layers.{i}" in getattr(
                model, "hf_device_map", {}
            ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                dev = model.hf_device_map[f"model.layers.{i}"]
                inps, outs, position_ids = (
                    inps.to(dev),
                    outs.to(dev),
                    position_ids.to(dev),
                )

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = BiasGPT(subset[name], args.metrics)

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                if name == "self_attn.o_proj":
                    W_metric = metrics[args.metrics](wrapped_layers, subset, name) ** 2
                    attn_metric_list.append(W_metric.cpu())
                    attn_baseline_inp_list.append(
                        wrapped_layers[name].baseline_inp.type(torch.half)
                    )
                else:
                    W_metric = metrics[args.metrics](wrapped_layers, subset, name)
                    mlp_metric_list.append(W_metric.cpu())
                    mlp_baseline_inp_list.append(
                        wrapped_layers[name].baseline_inp.type(torch.half)
                    )
                wrapped_layers[name].free()

            inps, outs = (
                outs,
                inps,
            )  # Use the original output as input to the next layer
            torch.cuda.empty_cache()

        standarlization = lambda x: (
            x - torch.mean(x, axis=1, keepdim=True)
        ) / torch.std(x, axis=1, keepdim=True)

        if args.structure in ["AL-AM"]:
            attn_metric = torch.stack(attn_metric_list)
            attn_metric = standarlization(attn_metric)
            attn_metric = attn_metric.reshape(
                len(layers) - args.start_pruning_layer_idx,
                -1,
                args.head_dim * args.gqa_groups,
            ).mean(dim=2)

            mlp_metric = torch.stack(mlp_metric_list)
            mlp_metric = standarlization(mlp_metric)

            prune_metric = torch.cat([attn_metric.view(-1), mlp_metric.view(-1)])
            sorted_prune, indices = torch.sort(prune_metric, descending=True)
            compression_weight = torch.ones_like(indices)
            compression_weight[indices < attn_metric.numel()] = 512.0 / 3
            threshold = sorted_prune[
                torch.argmin(
                    torch.abs(
                        torch.cumsum(compression_weight, 0)
                        - torch.sum(compression_weight) * (1 - args.pruning_ratio)
                    )
                )
            ]
            attn_mask = attn_metric > threshold
            mlp_mask = mlp_metric > threshold

        for idx in range(len(layers) - args.start_pruning_layer_idx):
            if f"model.layers.{i}" in getattr(model, "hf_device_map", {}):
                compress(
                    model.model.layers[idx],
                    attn_mask[idx],
                    None,
                    attn_baseline_inp_list[idx],
                    None,
                    model.hf_device_map[f"model.layers.{idx}"],
                    args=args,
                )
            else:
                compress(
                    model.model.layers[idx],
                    attn_mask[idx],
                    None,
                    attn_baseline_inp_list[idx],
                    None,
                    device,
                    args=args,
                )

            if f"model.layers.{i}" in getattr(model, "hf_device_map", {}):
                compress(
                    model.model.layers[idx],
                    None,
                    mlp_mask[idx],
                    None,
                    mlp_baseline_inp_list[idx],
                    model.hf_device_map[f"model.layers.{idx}"],
                )
            else:
                compress(
                    model.model.layers[idx],
                    None,
                    mlp_mask[idx],
                    None,
                    mlp_baseline_inp_list[idx],
                    device,
                )
            logger.info(f"[pruned] model.layers.{args.start_pruning_layer_idx + idx}")

        model.config.use_cache = use_cache
        torch.cuda.empty_cache()


class FlapPruner(TextGenerationAlgorithm):
    def __init__(self, job: LMJob, **kwargs):
        self.job = job
        self.args = create_instance(
            FlapConfig,
            {
                **kwargs,
                "_config_path": job.model.model_path,
            },
        )
        log_dict(asdict(self.args), "FlapConfig.")
        self.kw = kwargs
        self.output_dir = self.job.user_dir.output

        self.set_pruner(self.args, self.job, self.kw)
        self.adapter = None
        if self.kw.get("to_finetune", True):
            self.set_adapter()

    def set_adapter(self):
        """Factory method to set the adapter object."""

        # nyuntam-adapt
        from nyuntam_adapt.tasks import export_task_modules
        from nyuntam_adapt.utils.params_utils import AdaptParams

        CausalLLM = export_task_modules(str(AdaptTasks.TEXT_GENERATION))

        self.adapt_params = create_instance(AdaptParams, self.kw)
        # value updates
        self.adapt_params.DATASET_ARGS.CUSTOM_DATASET_PATH = str(
            self.job.dataset.dataset_name_or_path
        )
        self.adapt_params.DATASET_ARGS.FORMAT_STRING = self.job.dataset.format_string
        self.adapt_params.LOGGING_PATH = str(self.job.user_dir.logs)
        self.adapt_params.MODEL_ARGS.MODEL_PATH = self.job.model.model_name
        self.adapt_params.MODEL_ARGS.LOCAL_MODEL_PATH = str(
            (self.job.user_dir.output).absolute()
        )
        self.adapt_params.OUTPUT_DIR = str(self.job.user_dir.output.absolute())
        self.adapt_params.cuda_id = ",".join(
            map(str, self.job.environment.cuda_device_ids)
        )

        self.adapter = CausalLLM(**asdict(self.adapt_params))

    def set_pruner(self, args, job, kwargs):
        self.pruner = Pruner(args=args, job=job, kwargs=kwargs)

    def export_scripts(self):
        # check if quant(4bit/8bit)
        bnb_config = None
        if self.adapt_params.BNB_CONFIG.USE_4BIT.load_in_4bit:
            # loading in 4bit
            bnb_config = asdict(self.adapt_params.BNB_CONFIG.USE_4BIT)
        elif self.adapt_params.BNB_CONFIG.USE_8BIT.load_in_8bit:
            # loading in 8bit
            bnb_config = asdict(self.adapt_params.BNB_CONFIG.USE_8BIT)

        if bnb_config is not None:
            import json

            with open(self.job.user_dir.output / "bnbconfig.json", "w") as f:
                json.dump(bnb_config, f, indent=4)

        export_script = f"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import json
from pathlib import Path
CURR = Path(__file__).parent.absolute()
print(CURR)
# To load in quant (4bit/8bit)
bnb_path = CURR / "bnbconfig.json"
bnb_path = bnb_path if bnb_path.exists() else None

if bnb_path is not None:
    with open(bnb_path) as f:
        bnbconfig = BitsAndBytesConfig(**json.load(f))

if bnb_path:
    model = AutoModelForCausalLM.from_pretrained("{self.job.model.model_name}", state_dict=torch.load("merged_model_state_dict.pth", map_location="cpu"), ignore_mismatched_sizes=True, quantization_config=bnbconfig, device_map="cpu")
else:
    # loads without quant (if not finetuned with quant)
    model = AutoModelForCausalLM.from_pretrained("{self.job.model.model_name}", state_dict=torch.load("merged_model_state_dict.pth", map_location="cpu"), ignore_mismatched_sizes=True, device_map="cpu")

model.to("cuda:0")

# forward pass
tokenizer = AutoTokenizer.from_pretrained("{self.job.model.model_name}")
inputs = tokenizer("Question:\nWhat is the meaning of life?\nAnswer:", return_tensors="pt").input_ids.to(model.device)
outputs = model.generate(inputs, max_new_tokens=100)
print(tokenizer.batch_decode(outputs)[0])

"""
        with open(self.job.user_dir.output / "run.py", "w") as f:
            f.write(export_script)

    def compress_model(self):
        assert self.pruner is not None
        self.pruner.prune()
        del self.pruner
        free()

        if self.adapter is not None:
            logger.info("finetuning")
            self.adapter.adapt_model()
            self.export_scripts()
        del self.adapter
        free()
        return None, None
