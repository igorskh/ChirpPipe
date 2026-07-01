import librosa
import numpy as np

from chirps.chirp_node import ChirpNode
from chirps.cli_chirp import CLIChirp

import argparse
import matplotlib.pyplot as plt


class SonogramAudio(CLIChirp, ChirpNode):
    hop_length = 512
    n_fft = 2048
    sample_rate = 32000
    output_format = "png"

    def configure(self, input_config: dict):
        self.hop_length = input_config.get("hop_length", self.hop_length)
        self.n_fft = input_config.get("n_fft", self.n_fft)
        self.sample_rate = input_config.get("sample_rate", self.sample_rate)
        self.output_format = input_config.get(
            "output_format", self.output_format)

    def process(self, **kwargs) -> dict:
        path = kwargs.get("path")
        if path is None:
            raise ValueError("Missing 'path' argument for sonogram.")

        output_path = self.generate_sonogram(path)
        return {"output_path": output_path}

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="Generate MEL sonogram from audio files.")
        parser.add_argument(
            "path", type=str, help="Path to the input audio file.")
        parser.add_argument(
            "--hop_length", type=int, default=self.hop_length,
            help=f"Hop length for STFT (default: {self.hop_length}).")
        parser.add_argument(
            "--n_fft", type=int, default=self.n_fft,
            help=f"FFT size (default: {self.n_fft}).")
        parser.add_argument(
            "--sample_rate", type=int, default=self.sample_rate,
            help=f"Sample rate for audio processing (default: {self.sample_rate}).")
        parser.add_argument(
            "--output_format", type=str, default=self.output_format,
            help=f"Output image format (default: {self.output_format}).")
        return parser

    def process_cli(self, args: argparse.Namespace) -> None:
        self.configure({"hop_length": args.hop_length,
                        "n_fft": args.n_fft,
                        "sample_rate": args.sample_rate,
                        "output_format": args.output_format})
        self.generate_sonogram(args.path)

    def generate_sonogram(self, path: str) -> str:
        y, sr = librosa.load(path, sr=self.sample_rate, mono=True)

        if y.size == 0:
            raise ValueError(f"Audio file '{path}' is empty.")

        duration = len(y) / sr

        spec_mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=128
        )

        spec_db = librosa.power_to_db(spec_mel, ref=np.max)

        fig, ax = plt.subplots(figsize=(12, 8))
        img = ax.imshow(spec_db, aspect="auto", origin="lower",
                        extent=[0, duration, 0, librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)[-1]],
                        cmap="viridis")

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("MEL Sonogram")

        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label("dB")

        num_ticks = int(duration) + 1
        ax.set_xticks(range(num_ticks))
        ax.set_xticklabels(range(num_ticks))

        for i in range(num_ticks):
            x = i
            alpha = 0.3
            rect = plt.Rectangle(
                (x, 0),
                1,
                librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)[-1],
                alpha=alpha,
                color="white",
                linewidth=0
            )
            ax.add_patch(rect)

        output_path = path.replace(".wav", "_sonogram.png")
        plt.savefig(output_path, format=self.output_format, dpi=100)
        plt.close()

        return output_path


if __name__ == "__main__":
    sonogram_audio = SonogramAudio()

    parser = sonogram_audio.parse_args()
    args = parser.parse_args()
    sonogram_audio.process_cli(args)
