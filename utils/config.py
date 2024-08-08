from json import load, dump as json_dump
from yaml import safe_load, dump as yaml_dump
from typing import Dict, Union
from pathlib import Path


def load_config(config_path: Union[str, Path]) -> Dict:
    if isinstance(config_path, str):
        config_path = Path(config_path)

    if config_path.suffix == ".yaml":
        with open(config_path, "r") as f:
            config = safe_load(f)
    elif config_path.suffix == ".json":
        with open(config_path, "r") as f:
            config = load(f)
    else:
        raise ValueError(f"Unknown file type: {config_path.suffix}")

    return config


def dump_config(config: Dict, config_path: Union[str, Path]) -> Path:
    if isinstance(config_path, str):
        config_path = Path(config_path)
    
    if config_path.exists():
        print(f"Warning: {config_path} already exists. Overwriting.")
        config_path.unlink()

    if config_path.suffix == ".yaml":
        with open(config_path, "w") as f:
            yaml_dump(config, f)
    elif config_path.suffix == ".json":
        with open(config_path, "w") as f:
            json_dump(config, f)
    else:
        raise ValueError(f"Unknown file type: {config_path.suffix}")

    return config_path
