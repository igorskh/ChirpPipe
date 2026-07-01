import librosa
import numpy as np

from chirps.chirp_node import ChirpNode
from chirps.cli_chirp import CLIChirp

import argparse


class RmsBars(CLIChirp, ChirpNode):
    resolution = 0.5  # default resolution in seconds
    sample_rate = 32000  # default sample rate for audio processing

    def configure(self, input_config: dict):
        self.resolution = input_config.get("resolution", self.resolution)
        self.sample_rate = input_config.get("sample_rate", self.sample_rate)

    def process(self, **kwargs) -> dict:
        path = kwargs.get("path")
        if path is None:
            raise ValueError("Missing 'path' argument for rms_bars.")

        rms_dict = self.compute_rms_bars(path)
        return {"rms_bars": rms_dict}

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="Compute RMS values per time bar from audio files.")
        parser.add_argument(
            "path", type=str, help="Path to the input audio file.")
        parser.add_argument(
            "--resolution", type=float, default=self.resolution,
            help=f"Time resolution in seconds for bars (default: {self.resolution}).")
        parser.add_argument(
            "--sample_rate", type=int, default=self.sample_rate,
            help=f"Sample rate for audio processing (default: {self.sample_rate}).")
        return parser

    def process_cli(self, args: argparse.Namespace) -> None:
        self.configure({"resolution": args.resolution,
                        "sample_rate": args.sample_rate})
        print(self.compute_rms_bars(args.path))

    def compute_rms_bars(self, path: str) -> dict:
        y, sr = librosa.load(path, sr=self.sample_rate, mono=True)

        if y.size == 0:
            raise ValueError(f"Audio file '{path}' is empty.")

        duration = len(y) / sr

        if duration < self.resolution:
            raise ValueError(
                f"Audio duration {duration}s is less than resolution {self.resolution}s.")

        num_samples = len(y)
        samples_per_bar = int(self.resolution * sr)

        rms_dict = {}
        bar_start = 0

        while bar_start < num_samples:
            bar_end = min(bar_start + samples_per_bar, num_samples)
            bar_samples = y[bar_start:bar_end]

            rms = np.sqrt(np.mean(bar_samples ** 2))

            start_time = bar_start / sr
            end_time = bar_end / sr
            bar_key = f"{start_time:.1f}-{end_time:.1f}"

            rms_dict[bar_key] = round(rms, 4)

            bar_start = bar_end

        return rms_dict


if __name__ == "__main__":
    rms_bars = RmsBars()

    parser = rms_bars.parse_args()
    args = parser.parse_args()
    rms_bars.process_cli(args)
