from dataclasses import dataclass, is_dataclass, fields, MISSING
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

CONFIG_KEY_SEPARATOR = chr(46)  # ascii 46 => '.' (dot)


# region build utilities
def build_nested_dict(flat_dict: Dict[str, Any]) -> Dict:
    """Builds a nested dictionary from a flat dictionary.
    The keys in the flat dictionary are separated by a dot.
    e.g.
    ```python
    {"a.b.c": 1} => {"a": {"b": {"c": 1}}}
    ```

    Args:
        flat_dict (dict): A flat dictionary.

    Returns:
        dict: A nested dictionary.
    """
    nested_dict = {}
    for key, value in flat_dict.items():
        keys = key.split(CONFIG_KEY_SEPARATOR)
        temp_dict = nested_dict
        for k in keys[:-1]:
            temp_dict = temp_dict.setdefault(k, {})
        temp_dict[keys[-1]] = value
    return nested_dict


def deep_update(
    mapping: Dict[Any, Any], *updating_mappings: Dict[Any, Any]
) -> Dict[Any, Any]:
    """Deeply updates a mapping with other mappings. Taken from - [pydantic deep_update](https://github.com/pydantic/pydantic/blob/fd2991fe6a73819b48c906e3c3274e8e47d0f761/pydantic/utils.py#L200)

    Args:
        mapping (Dict[Any, Any]): The mapping to update.
        *updating_mappings (Dict[Any, Any]): The mappings to update with.

    Returns:
        Dict[Any, Any]: The updated mapping.
    """
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


def create_instance(data_class: dataclass, flat_dict: Dict[str, Any]):
    instance_args = {}
    for field in fields(data_class):
        field_name = field.name
        field_type = field.type
        field_value = flat_dict.get(field_name, MISSING)

        if is_dataclass(field_type):
            instance_args[field_name] = create_instance(field_type, flat_dict)
        else:
            if field_value is None or field_value == "" or field_value == MISSING:
                instance_args[field_name] = field.default
            else:
                instance_args[field_name] = field_value

    dc = data_class(**instance_args)
    return dc


# endregion


# region logging utilities
def log_dict(d: Dict, prefix: str = ""):
    """Logs a dictionary.

    Args:
        d (Dict): The dictionary to log.
        prefix (str, optional): The prefix to add to the log. Defaults to "".
    """
    for k, v in d.items():
        if isinstance(v, dict):
            log_dict(v, prefix=f"{prefix}{k}.")
        else:
            logger.info(f"{prefix}{k}: {v}")


# endregion
