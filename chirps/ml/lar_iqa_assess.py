
import argparse
import logging
import os

import torch
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms

from chirps.chirp_node import ChirpNode
from chirps.cli_chirp import CLIChirp
from chirps.ml.lar_iqa_model.mobilenet_merged import MobileNetMerged
from chirps.utils import init_default_logger


class LarIqaAssess(CLIChirp, ChirpNode):
    model = None
    device = "cpu"
    model_path = "models/AIM_Training_2branche_MLP-Head.pt"

    @staticmethod
    def preprocess_image(image_path, color_space, device):
        image = Image.open(image_path).convert('RGB')
        if color_space == "HSV":
            image = Image.fromarray(cv2.cvtColor(
                np.array(image), cv2.COLOR_RGB2HSV))
        elif color_space == "LAB":
            image = Image.fromarray(cv2.cvtColor(
                np.array(image), cv2.COLOR_RGB2LAB))
        elif color_space == "YUV":
            image = Image.fromarray(cv2.cvtColor(
                np.array(image), cv2.COLOR_RGB2YUV))

        transform_authentic = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
        transform_synthetic = transforms.Compose([
            transforms.CenterCrop((1280, 1280)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
        image_authentic = transform_authentic(image).unsqueeze(0).to(device)
        image_synthetic = transform_synthetic(image).unsqueeze(0).to(device)
        return image_authentic, image_synthetic

    def infer(self, image_authentic, image_synthetic):
        with torch.no_grad():
            output = self.model(image_authentic, image_synthetic)
            return output.item()

    def map_score_to_rating(self, score):
        if score < 0.2:
            return 1
        elif score < 0.4:
            return 2
        elif score < 0.6:
            return 3
        elif score < 0.8:
            return 4
        else:
            return 5

    def process(self, **kwargs) -> dict:
        input_image_path = kwargs.get("input")
        if not input_image_path:
            raise ValueError("Input image path is required.")

        if self.model is None:
            self.init_model(self.model_path)

        image_paths = []
        if os.path.isdir(input_image_path):
            supported_exts = {"jpeg", "jpg", "png"}
            image_paths = [
                os.path.join(input_image_path, f)
                for f in os.listdir(input_image_path)
                if os.path.isfile(os.path.join(input_image_path, f))
                and os.path.splitext(f)[1].lower().lstrip(".") in supported_exts
            ]
        else:
            image_paths = [input_image_path]

        result = self.process_batch(
            image_paths, as_rating=kwargs.get("as_rating", False))
        return {"results": result}

    def process_one(self, image_path: str, as_rating: bool = False) -> dict:
        if self.model is None:
            self.init_model(self.model_path)

        image_authentic, image_synthetic = self.preprocess_image(
            image_path, color_space="RGB", device=self.device)
        score = self.infer(image_authentic, image_synthetic)

        if as_rating:
            rating = self.map_score_to_rating(score)
            return {"score": score, "rating": rating}

        return {"score": score}

    def process_batch(self, image_paths: list, as_rating: bool = False) -> dict:
        if self.model is None:
            self.init_model(self.model_path)

        results = []
        for image_path in image_paths:
            res = self.process_one(image_path, as_rating=as_rating)
            if as_rating:
                results.append(
                    {"image_path": image_path, "score": res["score"], "rating": res["rating"]})
            else:
                results.append(
                    {"image_path": image_path, "score": res["score"]})

        return results

    def configure(self, input_config: dict):
        model_path = input_config.get("model_path")
        if model_path:
            self.model_path = model_path
            self.init_model(self.model_path)

    def process_cli(self, args) -> dict:
        input_image_path = getattr(args, "input", None)
        if not input_image_path:
            raise ValueError("Input image path is required.")

        as_rating = getattr(args, "as_rating", False)

        self.configure(vars(args))

        res = self.process(input=input_image_path, as_rating=as_rating)
        for r in res["results"]:
            print(f"Image: {r['image_path']}, LAR-IQA Score: {r['score']:.4f}")

    def parse_args(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="LAR-IQA Assess CLI")
        parser.add_argument(
            "input", type=str, help="Input image file path")
        parser.add_argument(
            "--model", type=str, default=self.model_path, help="Path to the LAR-IQA model file")
        parser.add_argument(
            "--as_rating", type=bool, default=False, help="Whether to interpret the score as a rating (True or False)")
        return parser

    def init_model(self, model_path: str):
        self.model = MobileNetMerged()
        self.model.load_state_dict(torch.load(
            model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        logging.info(f"LAR-IQA model loaded from {model_path}")

        return self.model


if __name__ == "__main__":
    init_default_logger()

    lar_iqa_assess = LarIqaAssess()
    lar_iqa_assess.process_cli(args=lar_iqa_assess.parse_args().parse_args())
