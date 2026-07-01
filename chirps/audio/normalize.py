import librosa

from chirps.chirp_node import ChirpNode
from chirps.cli_chirp import CLIChirp

import soundfile as sf
import numpy as np

import argparse


class NormalizeAudio(CLIChirp, ChirpNode):
    normalize_level = -1.0  # default normalization level in dBFS
    sample_rate = 32000  # default sample rate for audio processing

    def configure(self, input_config: dict):
        self.normalize_level = input_config.get(
            "normalize_level", self.normalize_level)
        self.sample_rate = input_config.get(
            "sample_rate", self.sample_rate)
        self.out_postfix = input_config.get(
            "out_postfix", "normalized")

    def process(self, **kwargs) -> dict:
        path = kwargs.get("path")
        if path is None:
            raise ValueError("Missing 'path' argument for normalization.")

        y_normalized, output_path = self.normalize_audio(path)
        return {"y_normalized": y_normalized, "output_path": output_path}

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="Normalize audio files to a specified dBFS level.")
        parser.add_argument(
            "path", type=str, help="Path to the input audio file.")
        parser.add_argument(
            "--level", type=float, default=self.normalize_level,
            help=f"Normalization level in dBFS (default: {self.normalize_level}).")
        parser.add_argument(
            "--sample_rate", type=int, default=self.sample_rate,
            help=f"Sample rate for audio processing (default: {self.sample_rate}).")
        parser.add_argument("--out_postfix", type=str, default="normalized",
                            help="Output postfix for normalized audio files (default: 'normalized').")
        return parser

    def process_cli(self, args: argparse.Namespace) -> None:
        self.configure({"normalize_level": args.level,
                       "sample_rate": args.sample_rate,
                       "out_postfix": args.out_postfix})
        self.normalize_audio(args.path)

    def normalize_audio(self, path: str, remove_dc: bool = True,
                        independent_channels: bool = False) -> str:
        """
        Audacity-style peak normalization.

        self.normalize_level: target peak level in dBFS (e.g. -1.0).
        remove_dc: subtract per-channel mean before scaling (matches
                Audacity's default "Remove DC offset" behavior).
        independent_channels: if True, each channel is scaled by its own
                factor. If False (Audacity default), all channels are
                scaled by the same factor, derived from the loudest channel.
        """
        y, sr = librosa.load(path, sr=self.sample_rate, mono=True)

        if y.size == 0:
            raise ValueError(f"Audio file '{path}' is empty.")

        is_mono = (y.ndim == 1)
        if is_mono:
            # treat as single-channel 2D for uniform logic
            y = y[:, np.newaxis]

        y = y.astype(np.float64)

        # Step 1: remove DC offset per channel
        if remove_dc:
            y = y - np.mean(y, axis=0, keepdims=True)

        # Step 2: find peak(s)
        target_linear = 10 ** (self.normalize_level / 20)

        if independent_channels:
            peaks = np.max(np.abs(y), axis=0, keepdims=True)
            peaks[peaks == 0] = 1.0  # avoid divide-by-zero on silent channels
            gain = target_linear / peaks
            y_normalized = y * gain
        else:
            peak = np.max(np.abs(y))
            gain = target_linear / peak if peak > 0 else 1.0
            y_normalized = y * gain

        if is_mono:
            y_normalized = y_normalized[:, 0]

        output_path = path.replace(".wav", f"_{self.out_postfix}.wav")
        sf.write(output_path, y_normalized, sr)

        return y_normalized, output_path


if __name__ == "__main__":
    normalize_audio = NormalizeAudio()

    parser = normalize_audio.parse_args()
    args = parser.parse_args()
    normalize_audio.process_cli(args)
