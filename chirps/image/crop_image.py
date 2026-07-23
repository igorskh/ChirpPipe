import argparse
from chirps.chirp_node import ChirpNode
from chirps.cli_chirp import CLIChirp

from PIL import Image
from typing import List, Tuple

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
        parser.add_argument(
            "--square", type=int, default=224,
            help="Square bounding box size.")
        parser.add_argument("--out_postfix", type=str, default="cropped",
                            help="Output postfix for cropped image files (default: 'cropped').")
        return parser

    def process_cli(self, args: argparse.Namespace) -> None:
        self.configure({
            "out_postfix": args.out_postfix
        })
        self.process(path=args.path, xyxy=args.xyxy, square=args.square)

    def configure(self, input_config: dict):
        self.out_postfix = input_config.get("out_postfix", "cropped")

    def process(self, **kwargs) -> dict:
        path = kwargs.get("path")
        if path is None:
            raise ValueError("Missing 'path' argument for cropping.")

        square_size = kwargs.get("square", 224)

        xyxy = kwargs.get("xyxy")
        if xyxy is None:
            raise ValueError("Missing 'xyxy' argument for cropping.")

        cropped, output_path = self.crop_image(path, xyxy, square_size)
        return {"cropped": cropped, "output_path": output_path}

    def crop_image(self, path: str, xyxy: List[float], square_size: int) -> Tuple[Image.Image, str]:
        """
        Crop image to a specified bounding box.

        xyxy: bounding box coordinates in the format [x1, y1, x2, y2].
        """
        if xyxy is None:
            raise ValueError(
                "Bounding box coordinates (xyxy) must be provided for cropping.")

        img = Image.open(path)

        x1, y1, x2, y2 = map(float, xyxy)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid 'xyxy' coordinates: x2/y2 must be greater than x1/y1.")

        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # Build a square crop around the bbox center. Keep at least square_size,
        # and keep larger bbox content by using the max side.
        side = max(float(square_size), bbox_w, bbox_h)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        square_x1 = int(round(cx - side / 2.0))
        square_y1 = int(round(cy - side / 2.0))
        square_x2 = int(round(square_x1 + side))
        square_y2 = int(round(square_y1 + side))

        # PIL supports out-of-bounds crop; it pads missing regions,
        # which lets us keep square_size even for small source images.
        cropped_img = img.crop((square_x1, square_y1, square_x2, square_y2))

        if cropped_img.size != (square_size, square_size):
            resample_filter = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
            cropped_img = cropped_img.resize((square_size, square_size), resample_filter)

        # Save the cropped image
        output_path = f"{path.rsplit('.', 1)[0]}_{self.out_postfix}.jpg"
        cropped_img.save(output_path)

        return cropped_img, output_path


if __name__ == "__main__":
    init_default_logger()

    crop_image = CropImage()
    cli_args = crop_image.parse_args()
    args = cli_args.parse_args()
    crop_image.process_cli(args)
