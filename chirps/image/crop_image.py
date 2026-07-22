import argparse
from chirps.chirp_node import ChirpNode
from chirps.cli_chirp import CLIChirp

from PIL import Image
from typing import List

from chirps.utils import init_default_logger


class CropImage(CLIChirp, ChirpNode):
    out_postfix = "cropped"

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="Crop image files to a specified bounding box.")
        parser.add_argument(
            "path", type=str, help="Path to the input image file.")
        parser.add_argument(
            "--xyxy", type=float, nargs=4, default=None,
            help="Bounding box coordinates in the format [x1, y1, x2, y2].")
        parser.add_argument("--out_postfix", type=str, default="cropped",
                            help="Output postfix for cropped image files (default: 'cropped').")
        return parser

    def process_cli(self, args: argparse.Namespace) -> None:
        self.configure({
            "out_postfix": args.out_postfix
        })
        self.process(path=args.path, xyxy=args.xyxy)

    def configure(self, input_config: dict):
        self.out_postfix = input_config.get("out_postfix", "cropped")

    def process(self, **kwargs) -> dict:
        path = kwargs.get("path")
        if path is None:
            raise ValueError("Missing 'path' argument for cropping.")

        xyxy = kwargs.get("xyxy")
        if xyxy is None:
            raise ValueError("Missing 'xyxy' argument for cropping.")

        cropped, output_path = self.crop_image(path, xyxy)
        return {"cropped": cropped, "output_path": output_path}

    def crop_image(self, path: str, xyxy: List[float]) -> str:
        """
        Crop image to a specified bounding box.

        xyxy: bounding box coordinates in the format [x1, y1, x2, y2].
        """
        if xyxy is None:
            raise ValueError(
                "Bounding box coordinates (xyxy) must be provided for cropping.")

        img = Image.open(path)

        # Crop the image using the bounding box
        cropped_img = img.crop(xyxy)

        # Save the cropped image
        output_path = f"{path.rsplit('.', 1)[0]}_{self.out_postfix}.png"
        cropped_img.save(output_path)

        return cropped_img, output_path


if __name__ == "__main__":
    init_default_logger()

    crop_image = CropImage()
    cli_args = crop_image.parse_args()
    args = cli_args.parse_args()
    crop_image.process_cli(args)
