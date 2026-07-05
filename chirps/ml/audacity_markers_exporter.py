import argparse
import logging
import sys
import pandas as pd

from chirps.chirp_node import ChirpNode
from chirps.cli_chirp import CLIChirp
from chirps.utils import init_default_logger


class AudacityMarkersExporter(CLIChirp, ChirpNode):
    """CLI tool to embed BirdNET detections as Audacity labels in a WAV file."""

    @staticmethod
    def create_label_text(row):
        return f"{row['label']} ({row['confidence']:.2f})"

    @staticmethod
    def create_audacity_labels(df):
        df["label"] = df.apply(
            AudacityMarkersExporter.create_label_text, axis=1)
        return df[["start_sec", "end_sec", "label"]]

    @staticmethod
    def save_audacity_labels(df, input_path, predictor_name) -> str:
        path_prefix = input_path.rsplit(".", 1)[0]

        df = AudacityMarkersExporter.create_audacity_labels(df)
        out_path = f"{path_prefix}_{predictor_name}.txt"

        df.to_csv(out_path, sep="\t", index=False, header=False)
        return out_path

    @staticmethod
    def load_detections(csv_path, min_confidence=0.0):
        df = pd.read_csv(csv_path)
        return df[df["confidence"] >= min_confidence]

    def configure(self, input_config: dict):
        pass

    def process(self, input_data: dict) -> dict:
        df = AudacityMarkersExporter.load_detections(
            input_data["csv_path"], min_confidence=input_data.get("min_confidence", 0.0))
        if df.empty:
            sys.exit("No detections found after filtering — nothing to write.")

        output_path = input_data.get(
            "output") or input_data["csv_path"].rsplit(".", 1)[0] + ".txt"

        AudacityMarkersExporter.save_audacity_labels(
            df, input_data["csv_path"], predictor_name="BirdNET")
        logging.info(f"Wrote {len(df)} label(s) to {output_path}")

        return {"output_path": output_path, "num_labels": len(df)}

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="Generate Audacity labels from BirdNET detections CSV file."
        )
        parser.add_argument("csv_path")
        parser.add_argument("-o", "--output")
        parser.add_argument("--min-confidence", type=float, default=0.0)
        return parser

    def process_cli(self, args) -> None:
        input_data = {
            "csv_path": args.csv_path,
            "output": args.output,
            "min_confidence": args.min_confidence,
        }
        self.process(input_data)


if __name__ == "__main__":
    init_default_logger()
    exporter = AudacityMarkersExporter()
    args = exporter.parse_args().parse_args()
    exporter.process_cli(args)
