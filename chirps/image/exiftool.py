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
        parser.add_argument(
            "--set_gps", type=str, default=None,
            help="Set the GPS coordinates for the image in the format 'latitude,longitude'.")
        parser.add_argument("--get_rating", action="store_true",
                            help="Get the rating for the image.")
        parser.add_argument("--get_metadata", action="store_true",
                            help="Get the metadata for the image.")
        parser.add_argument(
            "--get_gps", action="store_true",
            help="Get the GPS coordinates for the image.")
        return parser.parse_args()

    def process_cli(self, args: argparse.Namespace) -> None:
        self.configure({})
        res = self.process(
            path=args.path,
            set_rating=args.set_rating,
            get_rating=args.get_rating,
            set_gps=args.set_gps,
            get_gps=args.get_gps,
            get_metadata=args.get_metadata,
        )
        logging.info(f"ExifTool processing completed for {args.path}.")
        if args.get_rating:
            logging.info(f"Rating: {res['rating']}")
        if args.get_gps:
            logging.info(f"GPS: {res['gps']}")
        if args.get_metadata:
            logging.info(f"Metadata: {res['metadata']}")

    def configure(self, input_config: dict):
        pass

    def process(self, **kwargs) -> dict:
        set_rating = kwargs.get("set_rating")
        get_rating = kwargs.get("get_rating")
        set_gps = kwargs.get("set_gps")
        get_gps = kwargs.get("get_gps")
        path = kwargs.get("path")
        get_metadata = kwargs.get("get_metadata")

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

            if set_gps is not None:
                if set_gps == "":
                    et.execute("-GPSLatitude=", "-GPSLongitude=", path)
                    return {"metadata": et.get_metadata(path), "rating": None, "gps": None}

                try:
                    latitude, longitude = map(float, set_gps.split(","))
                except ValueError:
                    raise ValueError(
                        "GPS coordinates must be in the format 'latitude,longitude'.")
                et.execute(f"-GPSLatitude={latitude}",
                           f"-GPSLongitude={longitude}", path)

            metadata = et.get_metadata(path)
            if get_rating:
                rating = metadata[0].get("XMP:Rating")
            else:
                rating = None

            if get_gps:
                gps = metadata[0].get("EXIF:GPSLatitude"), metadata[0].get(
                    "EXIF:GPSLongitude")
            else:
                gps = None

        result = {}
        if get_metadata:
            result["metadata"] = metadata[0]
        if get_rating:
            result["rating"] = rating
        if get_gps:
            result["gps"] = gps

        return result


if __name__ == "__main__":
    init_default_logger()

    exif_tool = ExifTool()

    cli_args = exif_tool.parse_args()
    exif_tool.process_cli(cli_args)
