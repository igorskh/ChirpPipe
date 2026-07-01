import librosa

from chirps.chirp_node import ChirpNode
from chirps.cli_chirp import CLIChirp

import soundfile as sf

import argparse


class CropAudio(CLIChirp, ChirpNode):
    start_time = 0.0  # default start time in seconds
    end_time = None  # default end time (end of audio)
    sample_rate = 32000  # default sample rate for audio processing

    def configure(self, input_config: dict):
        self.start_time = input_config.get("start_time", self.start_time)
        self.end_time = input_config.get("end_time", self.end_time)
        self.sample_rate = input_config.get("sample_rate", self.sample_rate)
        self.out_postfix = input_config.get("out_postfix", "cropped")

    def process(self, **kwargs) -> dict:
        path = kwargs.get("path")
        if path is None:
            raise ValueError("Missing 'path' argument for cropping.")

        y_cropped, output_path = self.crop_audio(path)
        return {"y_cropped": y_cropped, "output_path": output_path}

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="Crop audio files to a specified time range.")
        parser.add_argument(
            "path", type=str, help="Path to the input audio file.")
        parser.add_argument(
            "--start_time", type=float, default=self.start_time,
            help=f"Start time in seconds (default: {self.start_time}).")
        parser.add_argument(
            "--end_time", type=float, default=self.end_time,
            help=f"End time in seconds (default: end of audio).")
        parser.add_argument(
            "--sample_rate", type=int, default=self.sample_rate,
            help=f"Sample rate for audio processing (default: {self.sample_rate}).")
        parser.add_argument("--out_postfix", type=str, default="cropped",
                            help="Output postfix for cropped audio files (default: 'cropped').")
        return parser

    def process_cli(self, args: argparse.Namespace) -> None:
        self.configure({"start_time": args.start_time,
                        "end_time": args.end_time,
                        "sample_rate": args.sample_rate,
                        "out_postfix": args.out_postfix})
        self.crop_audio(args.path)

    def crop_audio(self, path: str) -> str:
        """
        Crop audio to a specified time range.

        self.start_time: start time in seconds.
        self.end_time: end time in seconds (None means end of audio).
        """
        y, sr = librosa.load(path, sr=self.sample_rate, mono=True)

        if y.size == 0:
            raise ValueError(f"Audio file '{path}' is empty.")

        # Convert times to samples
        start_sample = int(self.start_time * sr)
        end_sample = None if self.end_time is None else int(self.end_time * sr)

        if start_sample >= len(y):
            raise ValueError(
                f"Start time {self.start_time}s exceeds audio duration.")

        y_cropped = y[start_sample:end_sample]

        if y_cropped.size == 0:
            raise ValueError(
                f"Crop range results in empty audio (start={self.start_time}s, end={self.end_time}s).")

        output_path = path.replace(".wav", f"_{self.out_postfix}.wav")
        sf.write(output_path, y_cropped, sr)

        return y_cropped, output_path


if __name__ == "__main__":
    crop_audio = CropAudio()

    parser = crop_audio.parse_args()
    args = parser.parse_args()
    crop_audio.process_cli(args)
