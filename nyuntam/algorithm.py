from abc import ABC, ABCMeta, abstractmethod

__all__ = [
    "Algorithm",
    "CompressionAlgorithm",
    "TextGenerationAlgorithm"
]


class Algorithm(ABC):
    """Base class for all of Nyun's algorithms."""

    @abstractmethod
    def run(self):
        """The run method for the algorithm."""


class CompressionAlgorithm(Algorithm, metaclass=ABCMeta):
    """Base class for all of Nyun's compression algorithms."""

    def run(self):
        """The run method for the algorithm."""
        self.compress_model()

    @abstractmethod
    def compress_model(self):
        """Compress the model."""


class TextGenerationAlgorithm(CompressionAlgorithm):
    """Base class for all of Nyun's Text Generation compression algorithms."""
