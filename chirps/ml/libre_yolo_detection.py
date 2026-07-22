
import argparse

from chirps.chirp_node import ChirpNode
from chirps.cli_chirp import CLIChirp
from chirps.utils import init_default_logger
from libreyolo import LibreYOLO


class LibreYoloDetection(CLIChirp, ChirpNode):
    model = None
    model_path = "models/LibreYOLO9t.pt"

    def process(self, **kwargs) -> dict:
        input_image_path = kwargs.get("input")
        if not input_image_path:
            raise ValueError("Input image path is required.")

        if self.model is None:
            self.init_model(self.model_path)

        detections = self.model(input_image_path)

        return {"detections": detections}

    def configure(self, input_config: dict):
        model_path = input_config.get("model_path")
        if model_path:
            self.model_path = model_path
            self.init_model(self.model_path)

    def process_cli(self, args) -> dict:
        input_image_path = getattr(args, "input", None)
        if not input_image_path:
            raise ValueError("Input image path is required.")

        self.configure(vars(args))

        res = self.process(input=input_image_path)

        for b in res["detections"].boxes[0]:
            print(
                f"Detected class: {self.model.names[int(b.cls)]} [{float(b.conf):.2f}] Bounding box: {b.xyxy}")

    def parse_args(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="LibreYOLO Detection CLI")
        parser.add_argument(
            "input", type=str, help="Input image file path")
        parser.add_argument(
            "--model", type=str, default=self.model_path, help="Path to the LibreYOLO model file")
        return parser

    def init_model(self, model_path: str):
        self.model = LibreYOLO(model_path)


if __name__ == "__main__":
    init_default_logger()

    yolo_detection = LibreYoloDetection()
    parser = yolo_detection.parse_args()
    args = parser.parse_args()
    yolo_detection.process_cli(args)
