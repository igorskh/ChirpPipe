
import argparse
import logging
import os
import json
import pandas as pd

from chirps.chirp_node import ChirpNode
from chirps.cli_chirp import CLIChirp


from bioclip import Rank
from bioclip.predict import TreeOfLifeClassifier

from chirps.utils import init_default_logger


class BioClipInference(CLIChirp, ChirpNode):
    model = None
    gradcam_prefix = "gradcam_"
    threshold = 0.2

    def process(self, input_data: dict) -> pd.DataFrame:
        image_path = input_data.get("image_path")
        threshold = input_data.get("threshold", self.threshold)
        output_format = input_data.get("output_format", "json")

        if not image_path:
            logging.error("Image path must be provided.")
            return pd.DataFrame()

        if os.path.isdir(image_path):
            logging.info(f"Processing directory: {image_path}")
            supported_exts = {"jpeg", "jpg", "png"}
            files = [
                os.path.join(image_path, f)
                for f in os.listdir(image_path)
                if os.path.isfile(os.path.join(image_path, f))
                and os.path.splitext(f)[1].lower().lstrip(".") in supported_exts
                and not f.startswith(self.gradcam_prefix)
            ]
            res = self.predict_batch(files)
        else:
            res = self.predict(image_path)

        df = self.create_dataframe(res, threshold=threshold)

        image_folder = image_path if os.path.isdir(
            image_path) else os.path.dirname(image_path)
        output_filename = f"{'' if os.path.isdir(
            image_path) else os.path.basename(image_path).split('.')[0]}_bioclip2.{output_format.lower()}"
        if output_format == "json":
            output_file = os.path.join(image_folder, output_filename)
            self.save_json(df, output_file, threshold=threshold)
        elif output_format == "csv":
            output_file = os.path.join(image_folder, output_filename)
            self.save_csv(df, output_file, threshold=threshold)

        return {
            "output_path": output_file,
            "predictions": df.to_dict(orient="records")
        }

    def configure(self, input_config: dict):
        pass

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="BioClip Inference CLI")
        parser.add_argument("image_path", type=str,
                            help="Path to the input image or directory.")
        parser.add_argument("--threshold", type=float, default=self.threshold,
                            help="Threshold for prediction confidence (default: 0.2).")
        parser.add_argument("--output_format", type=str, default="json",
                            help="Output format for predictions (default: 'json').")
        return parser

    def create_dataframe(self, predictions, threshold=None):
        df = pd.DataFrame(predictions)
        if threshold is not None:
            df = df[df['score'] >= threshold]
        return df

    def save_json(self, df: pd.DataFrame, output_file, threshold=None):
        with open(output_file, "w") as f:
            json.dump(df.to_dict(orient="records"), f, indent=4)
        logging.info(f"Predictions saved to {output_file}")

    def save_csv(self, df: pd.DataFrame, output_file, threshold=None):
        df.to_csv(output_file, index=False)
        logging.info(f"Predictions saved to {output_file}")

    def process_cli(self, args):
        res = self.process({
            "image_path": args.image_path,
            "threshold": args.threshold,
            "output_format": args.output_format
        })

        if not res["predictions"]:
            logging.info("No predictions above the threshold.")
            return

        for p in res["predictions"]:
            if p['score'] < args.threshold:
                continue
            file_name = p['file_name'].split(os.sep)[-1]
            logging.info(
                f"{file_name}: {p['common_name']} ({p['species']}) [{p['score']:.2f}]")

    def load_model(self):
        self.model = TreeOfLifeClassifier()
        logging.info("Model loaded successfully.")

    def predict_batch(self, image_paths):
        if self.model is None:
            self.load_model()

        return self.model.predict(image_paths, Rank.SPECIES)

    def predict(self, image_path):
        if self.model is None:
            self.load_model()

        result = self.model.predict(image_path, Rank.SPECIES)
        return result


if __name__ == "__main__":
    init_default_logger()

    bio_clip_inference = BioClipInference()
    parser = bio_clip_inference.parse_args()
    args = parser.parse_args()
    bio_clip_inference.process_cli(args)
