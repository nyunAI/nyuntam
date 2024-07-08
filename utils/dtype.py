import torch


def get_dtype_from_string(dtype: str) -> torch.dtype:
    """Get a torch dtype from a string."""
    return eval(f"torch.{dtype}")
