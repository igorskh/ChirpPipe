import argparse
import logging
import os
from pathlib import Path

from matplotlib import image as mpimg
import rawpy

from chirps.chirp_node import ChirpNode
from chirps.cli_chirp import CLIChirp

from chirps.utils import init_default_logger

SUPPORTED_RAW_EXTENSIONS = {".CR2", ".NEF", ".ARW", ".ORF", ".RW2", ".DNG"}


class RawProcessor(CLIChirp, ChirpNode):
    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="Run RawProcessor on image files.")
        parser.add_argument(
            "path", type=str, help="Path to the input image file or directory.")
        parser.add_argument("--generate_preview", action="store_true",
                            help="Generate a preview for the image.")
        return parser.parse_args()

    def process_cli(self, args: argparse.Namespace) -> None:
        self.configure({})
        res = self.process(
            path=args.path,
            generate_preview=args.generate_preview,
        )
        logging.info(f"RawProcessor processing completed for {args.path}.")

    def configure(self, input_config: dict):
        pass

    def process(self, **kwargs) -> dict:
        path = kwargs.get("path")
        generate_preview = kwargs.get("generate_preview", False)

        if path is None:
            raise ValueError(
                "Missing 'path' argument for RawProcessor processing.")

        if os.path.isdir(path):
            results = []
            for entry in sorted(os.listdir(path)):
                file_path = os.path.join(path, entry)
                if not os.path.isfile(file_path):
                    continue
                if Path(file_path).suffix.upper() not in SUPPORTED_RAW_EXTENSIONS:
                    continue
                try:
                    results.append(self.process_one(
                        file_path, generate_preview=generate_preview))
                except rawpy.LibRawFileUnsupportedError:
                    # Skip unsupported RAW files.
                    continue

            return {"status": "success", "path": path, "results": results}

        return self.process_one(path, generate_preview=generate_preview)

    def process_one(self, path: str, generate_preview: bool = False) -> dict:
        preview_path = None

        if generate_preview:
            logging.info(f"Generating preview for {path}...")
            try:
                with rawpy.imread(path) as raw:
                    rgb = raw.postprocess(use_camera_wb=True)
            except rawpy.LibRawFileUnsupportedError as exc:
                logging.error(f"Unsupported RAW file: {path}")
                return {"status": "error", "path": path, "error": str(exc)}

            source = Path(path)
            preview_path = str(source.with_name(f"{source.stem}_preview.jpg"))
            mpimg.imsave(preview_path, rgb)
            logging.info(f"Preview image saved to {preview_path}")

        return {"status": "success", "path": path, "preview_path": preview_path}


if __name__ == "__main__":
    init_default_logger()

    raw_processor = RawProcessor()
    args = raw_processor.parse_args()
    raw_processor.process_cli(args)
