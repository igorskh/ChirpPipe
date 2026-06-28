from abc import ABC, abstractmethod

class CLIChirp(ABC):
    """Interface for CLI chirp."""

    @abstractmethod
    def parse_args(self):
        """Return an argparse.ArgumentParser for the CLI."""
        pass

    @abstractmethod
    def process_cli(self, path: str, **kwargs) -> None:
        """Load a file for text extraction. Accepts all arguments from argparse."""