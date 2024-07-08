from abc import ABC, abstractmethod


class Algorithm(ABC):
    """Base class for all compression algorithms."""

    @abstractmethod
    def compress_model(self):
        """Compress the model."""
