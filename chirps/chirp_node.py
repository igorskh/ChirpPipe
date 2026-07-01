from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

# define possible return types for chirp nodes
ChirpNodeReturnType = dict | list | str | None | np.ndarray | pd.DataFrame | bool

class ChirpNode(ABC):
    """Interface for CLI chirp."""

    @abstractmethod
    def configure(self, input_config: dict):
        """Configure the chirp node with the given input configuration."""
        pass

    @abstractmethod
    def process(self, **kwargs) -> ChirpNodeReturnType:
        """Load a file for text extraction. Accepts all arguments from argparse."""
        pass