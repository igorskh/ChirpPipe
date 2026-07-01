from abc import ABC, abstractmethod
import argparse

class CLIChirp(ABC):
    """Interface for CLI chirp."""

    @abstractmethod
    def parse_args(self):
        """Return an argparse.ArgumentParser for the CLI."""
        pass

    @abstractmethod
    def process_cli(self, args: argparse.Namespace) -> None:
        """Load a file for text extraction. Accepts all arguments from argparse."""