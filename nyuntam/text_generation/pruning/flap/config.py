from dataclasses import dataclass
from transformers import AutoConfig
from typing import Optional, Union
from pathlib import Path


@dataclass
class FlapConfig:

    # add all the arguments here in sorted order
    eval: bool = False
    gqa_groups: Optional[int] = None
    head_dim: Optional[int] = None
    hidden_dim: Optional[int] = None
    metrics: str = "WIFV"
    nsamples: int = 1024
    pruning_ratio: float = 0.2
    remove_heads: int = -1
    seed: int = 0
    start_pruning_layer_idx: int = 22
    structure: str = "AL-AM"

    _config: Optional[AutoConfig] = None
    _config_path: Optional[Union[str, Path]] = None

    def __post_init__(self):
        assert self.metrics in [
            "IFV",
            "WIFV",
            "WIFN",
            "N/A",
        ], f"Invalid metrics: {self.metrics}. Supported metrics are ['IFV', 'WIFV', 'WIFN', 'N/A']"
        assert (
            self.pruning_ratio >= 0 and self.pruning_ratio <= 1
        ), f"Invalid pruning_ratio: {self.pruning_ratio}. It should be in (0, 1)"
        assert (
            self.remove_heads >= -1
        ), f"Invalid remove_heads: {self.remove_heads}. It should be greater than or equal to -1"
        assert self.structure in [
            "AL-AM"
        ], f"Invalid structure: {self.structure}. Supported structures are ['AL-AM']"
        assert (
            self.start_pruning_layer_idx >= 0
        ), f"Invalid start_pruning_layer_idx: {self.start_pruning_layer_idx}. It should be greater than or equal to 0"

        # set head_dim, hidden_dim, and gqa_groups if not provided
        if self.hidden_dim is None or self.head_dim is None or self.gqa_groups is None:
            assert (
                self._config is not None or self._config_path is not None
            ), "Either _config or _config_path should be set to infer head_dim, hidden_dim, and gqa_groups"
            not_none_else_default = lambda x, default: x if x is not None else default
            self._config = not_none_else_default(
                self._config, AutoConfig.from_pretrained(self._config_path)
            )
            self.hidden_dim = not_none_else_default(
                self.hidden_dim, self._config.hidden_size
            )  # 4096 (for llama3)
            self.gqa_groups = not_none_else_default(
                self.gqa_groups,
                (self._config.num_attention_heads // self._config.num_key_value_heads),
            )  # 32 // 8 = 4 (for llama3)
            self.head_dim = not_none_else_default(
                self.head_dim,
                (self._config.hidden_size // self._config.num_attention_heads),
            )  # 4096 // 32 = 128 (for llama3)
