import argparse
import logging
import os
from pathlib import Path

from chirps.chirp_node import ChirpNode
from chirps.cli_chirp import CLIChirp
import ollama

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class LLMVisionDescription(CLIChirp, ChirpNode):
    model_name = "llava:latest"
    embedding_model_name = "mxbai-embed-large"
    prompt_template = """Describe subjects and scene in this image in short.
Be concise, factual, and literal.
Do not mention uncertainty, introductions, or filler words.
Then output 3-5 comma-separated tags for the most important visible subjects, objects, setting, colors, and actions.
Format:
Description: [DESCRIPTION]
Tags: [TAG_1], [TAG_2], [TAG_3]"""

    prompt_with_species_pre_template = """Bird in this photo is %s."""

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="Run LLMVisionDescription on image files.")
        parser.add_argument(
            "path", type=str, help="Path to the input image file or directory.")
        parser.add_argument("--model_name", type=str, default="llava:latest",
                            help="Specify the model name to use.")
        parser.add_argument("--embedding_model_name", type=str, default="mxbai-embed-large",
                            help="Specify the embedding model name to use.")
        parser.add_argument("--species", type=str, default="",
                            help="Specify the species present in the image.")
        parser.add_argument("--return_embeddings", action="store_true",
                            help="Return text embeddings for the description.")
        return parser.parse_args()

    def process_cli(self, args: argparse.Namespace) -> None:
        self.configure({
            "model_name": args.model_name,
            "embedding_model_name": args.embedding_model_name
        })
        res = self.process(
            path=args.path,
            species=args.species,
            return_embeddings=args.return_embeddings
        )
        logging.info(
            f"LLMVisionDescription processing completed for {args.path}.")

        for result in res.get("results", []):
            print(f"Image: {result.get('image_path')}")
            print(f"{result.get('description').strip()}")

    def configure(self, input_config: dict):
        model_name = input_config.get("model_name", self.model_name)
        embedding_model_name = input_config.get(
            "embedding_model_name", self.embedding_model_name)

        self.model_name = model_name
        self.embedding_model_name = embedding_model_name

    def process(self, **kwargs) -> dict:
        path = kwargs.get("path")
        species = kwargs.get("species", "")
        return_embeddings = kwargs.get("return_embeddings", False)

        if path is None:
            raise ValueError(
                "Missing 'path' argument for LLMVisionDescription processing.")

        files = []
        if os.path.isfile(path):
            files.append(path)
        elif os.path.isdir(path):
            for entry in sorted(os.listdir(path)):
                file_path = os.path.join(path, entry)
                if not os.path.isfile(file_path):
                    continue
                if Path(file_path).suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                    continue
                files.append(file_path)

        if files:
            results = []
            for file_path in files:
                res = self.inference(file_path, species=species)
                emb = self.text_embedding(res) if return_embeddings else None
                results.append({
                    "image_path": file_path,
                    "description": res,
                    "embedding": emb
                })
            return {"results": results}
        else:
            raise ValueError(f"Invalid path: {path}")

    def inference(self, image_path: str, species: str = "") -> str:
        caption = ollama.chat(
            model=self.model_name,
            messages=[{
                'role': 'user',
                'content': (self.prompt_with_species_pre_template % species) + self.prompt_template if species else self.prompt_template,
                'images': [image_path]
            }]
        )['message']['content']
        return caption

    def text_embedding(self, text: str) -> list:
        res = ollama.embed(
            model=self.embedding_model_name,
            input=text
        )
        return res.embeddings[0]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    chirp = LLMVisionDescription()
    args = chirp.parse_args()
    chirp.process_cli(args)
