from transformers import AutoModelForCausalLM
import torch.nn as nn
from safetensors import safe_open
import glob
import torch
import json
from tqdm import tqdm
import os


def load_and_replace_weights(
    base_model_path, pruned_weights_path, cache_dir=None, overwrite_config=False
):
    assert base_model_path, "base_model_path not provided"
    assert os.path.exists(
        pruned_weights_path
    ), f"pruned_weights_path not found at {pruned_weights_path}"
    assert os.path.exists(
        f"{pruned_weights_path}/prune_config.json"
    ), f"prune_config.json not found at {pruned_weights_path}"

    if os.path.exists(f"{pruned_weights_path}/config.json"):
        config_path = f"{pruned_weights_path}/config.json"
    else:
        config_path = None

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
    )

    if len(glob.glob(f"{pruned_weights_path}/*.safetensors")) > 0:
        shapes = {}
        for tensor in glob.glob(f"{pruned_weights_path}/*.safetensors"):
            with safe_open(tensor, framework="pt") as f:
                for k in f.keys():
                    shapes[k] = f.get_tensor(k)
    else:
        # .pt outputs from pruning
        assert os.path.exists(
            f"{pruned_weights_path}/wds.pt"
        ), f"wds.pt not found at {pruned_weights_path}"
        shapes = torch.load(f"{pruned_weights_path}/wds.pt").state_dict()

    with open(f"{pruned_weights_path}/prune_config.json", "r") as f:
        prune_config = json.load(f)

    start_pruning_layer_idx = prune_config.get("start_pruning_layer_idx", 0)

    for layer_idx in tqdm(range(len(model.model.layers)), desc="Replacing weights"):
        if layer_idx >= start_pruning_layer_idx:
            for layer in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                curr_shape = (
                    model.model.layers[layer_idx].self_attn._modules[layer].weight.shape
                )
                pruned_shape = shapes.get(
                    f"model.layers.{layer_idx}.self_attn.{layer}.weight"
                )
                if pruned_shape is not None and curr_shape != pruned_shape.shape:
                    device = (
                        model.model.layers[layer_idx]
                        .self_attn._modules[layer]
                        .weight.device
                    )
                    bias = f"model.layers.{layer_idx}.self_attn.{layer}.bias" in shapes
                    model.model.layers[layer_idx].self_attn._modules[layer] = nn.Linear(
                        pruned_shape.shape[1],
                        pruned_shape.shape[0],
                        device=device,
                        bias=bias,
                    )
                    model.model.layers[layer_idx].self_attn._modules[
                        layer
                    ].weight.data = pruned_shape.to(device)
                    if bias:
                        model.model.layers[layer_idx].self_attn._modules[
                            layer
                        ].bias.data = shapes[
                            f"model.layers.{layer_idx}.self_attn.{layer}.bias"
                        ].to(
                            device
                        )
                del pruned_shape
                del curr_shape

            for layer in ["down_proj", "up_proj", "gate_proj"]:
                curr_shape = (
                    model.model.layers[layer_idx].mlp._modules[layer].weight.shape
                )
                pruned_shape = shapes.get(
                    f"model.layers.{layer_idx}.mlp.{layer}.weight"
                )
                if pruned_shape is not None and curr_shape != pruned_shape.shape:
                    device = (
                        model.model.layers[layer_idx].mlp._modules[layer].weight.device
                    )
                    bias = f"model.layers.{layer_idx}.mlp.{layer}.bias" in shapes
                    model.model.layers[layer_idx].mlp._modules[layer] = nn.Linear(
                        pruned_shape.shape[1],
                        pruned_shape.shape[0],
                        device=device,
                        bias=bias,
                    )
                    model.model.layers[layer_idx].mlp._modules[layer].weight.data = (
                        pruned_shape.to(device)
                    )
                    if bias:
                        model.model.layers[layer_idx].mlp._modules[layer].bias.data = (
                            shapes[f"model.layers.{layer_idx}.mlp.{layer}.bias"].to(
                                device
                            )
                        )
                del pruned_shape
                del curr_shape

        # Update number of heads and intermediate size
        model.model.layers[layer_idx].self_attn.num_heads = (
            model.model.layers[layer_idx]
            .self_attn._modules["q_proj"]
            .weight.data.shape[0]
            // prune_config["head_dim"]
        )
        model.model.layers[layer_idx].self_attn.num_key_value_heads = (
            model.model.layers[layer_idx]
            .self_attn._modules["k_proj"]
            .weight.data.shape[0]
            // prune_config["head_dim"]
        )
        model.model.layers[layer_idx].mlp.intermediate_size = (
            model.model.layers[layer_idx].mlp._modules["gate_proj"].weight.data.shape[0]
        )

    # Update the config file with the new model parameters
    if config_path:
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = model.config.to_dict()

    config["intermediate_size"] = [
        model.model.layers[layer_idx].mlp.intermediate_size
        for layer_idx in range(len(model.model.layers))
    ]
    config["num_attention_heads"] = [
        model.model.layers[layer_idx].self_attn.num_heads
        for layer_idx in range(len(model.model.layers))
    ]
    config["num_key_value_heads"] = [
        model.model.layers[layer_idx].self_attn.num_key_value_heads
        for layer_idx in range(len(model.model.layers))
    ]
    config["first_compressed_layer_idx"] = start_pruning_layer_idx
    config["auto_map"] = {
        "AutoConfig": "configuration_llama.LlamaConfig",
        "AutoModelForCausalLM": "modeling_llama.LlamaForCausalLM",
    }

    if overwrite_config:
        with open(config_path, "w") as outfile:
            json.dump(config, outfile, indent=2)

    model.config = model.config.from_dict(config)
    del shapes
    return model
