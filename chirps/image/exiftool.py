import argparse
import logging
from chirps.chirp_node import ChirpNode
from chirps.cli_chirp import CLIChirp

from chirps.utils import init_default_logger
from exiftool import ExifToolHelper


class ExifTool(CLIChirp, ChirpNode):
    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="Run ExifTool on image files.")
        parser.add_argument(
            "path", type=str, help="Path to the input image file.")
        parser.add_argument(
            "--set_rating", type=int, default=None,
            help="Set the rating for the image (0-5). 0 removes the rating.")
        parser.add_argument("--get_rating", action="store_true",
                            help="Get the rating for the image.")
        return parser.parse_args()

    def process_cli(self, args: argparse.Namespace) -> None:
        self.configure({})
        res = self.process(
            path=args.path,
            set_rating=args.set_rating,
            get_rating=args.get_rating
        )
        logging.info(f"ExifTool processing completed for {args.path}.")
        if args.get_rating:
            logging.info(f"Rating: {res['rating']}")

    def configure(self, input_config: dict):
        pass

    def process(self, **kwargs) -> dict:
        set_rating = kwargs.get("set_rating")
        get_rating = kwargs.get("get_rating")
        path = kwargs.get("path")
        if path is None:
            raise ValueError(
                "Missing 'path' argument for ExifTool processing.")

        with ExifToolHelper() as et:
            if set_rating is not None:
                if not (0 <= set_rating <= 5):
                    raise ValueError(
                        "Rating must be an integer between 0 and 5.")

                rating_arg = "-Rating=" if set_rating == 0 else f"-Rating={set_rating}"
                et.execute(rating_arg, path)
            metadata = et.get_metadata(path)
            if get_rating:
                rating = metadata[0].get("XMP:Rating")
            else:
                rating = None

        return {"metadata": metadata, "rating": rating}


if __name__ == "__main__":
    init_default_logger()

    exif_tool = ExifTool()

    cli_args = exif_tool.parse_args()
    exif_tool.process_cli(cli_args)
