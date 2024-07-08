import os
import logging
from typing import List

logger = logging.getLogger(__name__)


class CudaDeviceEnviron:
    _instance = None

    def __new__(cls, cuda_device_ids_str: str) -> "CudaDeviceEnviron":
        if cls._instance is None:
            cls._instance = super(CudaDeviceEnviron, cls).__new__(cls)
            cls._instance._cuda_device_ids = cls._instance._parse_cuda_device_ids(
                cuda_device_ids_str
            )
            cls._instance._set_cuda_visible_devices()
        else:
            logger.warn(
                f"Warning: CudaDeviceEnviron is a singleton class. Use CudaDeviceEnviron.get_instance() to get the existing instance."
            )
        return cls._instance

    @classmethod
    def get_instance(cls, cuda_device_ids_str: str = None) -> "CudaDeviceEnviron":
        if cls._instance is None:
            if cuda_device_ids_str is None:
                raise ValueError(
                    "cuda_device_ids_str must be provided when creating the first instance of CudaDeviceEnviron."
                )
            cls._instance = CudaDeviceEnviron(cuda_device_ids_str)
        return cls._instance

    @property
    def cuda_device_ids(self) -> List[int]:
        return list(map(str, range(self.num_device)))

    @property
    def available_cuda_devices(self) -> int:
        import torch

        return torch.cuda.device_count()

    @property
    def num_device(self) -> int:
        return len(self._cuda_device_ids)

    def _parse_cuda_device_ids(self, cuda_device_ids_str: str) -> List[int]:
        cuda_device_ids = [
            int(dev_id.strip()) for dev_id in cuda_device_ids_str.split(",")
        ]
        if not cuda_device_ids:
            raise ValueError(
                "The input string must contain at least one CUDA device ID."
            )
        return cuda_device_ids

    def _set_cuda_visible_devices(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self._cuda_device_ids))
        logger.info(f'Using devices: {os.environ.get("CUDA_VISIBLE_DEVICES")}')
