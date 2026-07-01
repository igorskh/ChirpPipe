
import csv
import logging
import os

import pandas as pd
from typing import List, Tuple, Optional

import librosa
import onnxruntime as ort
import numpy as np
import argparse

from chirps.chirp_node import ChirpNode
from chirps.cli_chirp import CLIChirp
from chirps.utils import init_default_logger

from .utils import download_file, chunk_audio


DEFAULT_MODEL_PATH = "models/birdnet_v3.onnx"
DEFAULT_LABELS_PATH = "models/birdnet_v3_labels.csv"
DEFAULT_MODEL_URL = "https://zenodo.org/records/20703646/files/BirdNET+_V3.0-preview3.1_Global_11K_FP32.onnx?download=1"
DEFAULT_LABELS_URL = "https://zenodo.org/records/20703646/files/BirdNET+_V3.0-preview3.1_Global_11K_Labels.csv?download=1"


class ONNXBioacousticsPredictor(CLIChirp, ChirpNode):
    labels = []
    chunk_length = 3.0  # default chunk length in seconds
    overlap = 0.0  # default overlap in seconds
    device = "cpu"  # default device
    session: "ort.InferenceSession" = None  # ONNX Runtime session
    predictions_key = "predictions"
    embeddings_key = "embeddings"
    label_format = "{sci_name}_{com_name}"
    out_postfix = "onnx_results"
    sampling_rate = 32000  # default sampling rate for audio processing
    min_conf = 0.2  # default minimum confidence threshold for predictions
    supported_exts = "wav,mp3,m4a,aac,flac,ogg"  # default supported file extensions
    model_path = DEFAULT_MODEL_PATH
    labels_path = DEFAULT_LABELS_PATH

    def process(self, **kwargs) -> dict:
        audio_path = kwargs.get("path")
        audio_samples = kwargs.get("samples")

        if audio_path is None and audio_samples is None:
            raise ValueError(
                "Missing 'path' or 'samples' argument for prediction.")

        df, embeddings, out_path = self.predict(
            kwargs.get("path"))

        return {"dataframe": df, "embeddings": embeddings, "out_path": out_path}

    def configure(self, input_config: dict):
        self.device = input_config.get("device", self.device)
        self.model_path = input_config.get("model", DEFAULT_MODEL_PATH)
        self.labels_path = input_config.get("labels", DEFAULT_LABELS_PATH)
        self.predictions_key = input_config.get(
            "predictions_key", self.predictions_key)
        self.embeddings_key = input_config.get(
            "embeddings_key", self.embeddings_key)
        self.label_format = input_config.get(
            "label_format", self.label_format)
        self.out_postfix = input_config.get("out_postfix", self.out_postfix)
        self.sampling_rate = input_config.get(
            "sampling_rate", self.sampling_rate)
        self.chunk_length = input_config.get("chunk_length", self.chunk_length)
        self.overlap = input_config.get("overlap", self.overlap)
        self.min_conf = input_config.get("min_conf", 0.2)
        self.supported_exts = input_config.get(
            "supported_exts", self.supported_exts)

    def parse_args(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="BirdNetV3 Predictor CLI")
        parser.add_argument("path", type=str,
                            help="Path to the audio file or folder to process.")
        parser.add_argument("--chunk_length", type=float, default=3.0,
                            help="Length of each audio chunk in seconds (default: 3.0).")
        parser.add_argument("--overlap", type=float, default=0.0,
                            help="Overlap between consecutive chunks in seconds (default: 0.0).")
        parser.add_argument("--min_conf", type=float, default=0.2,
                            help="Minimum confidence threshold for predictions (default: 0.2).")
        parser.add_argument("--device", type=str, choices=["cpu", "coreml"], default="cpu",
                            help="Device to run inference on (default: cpu). Use 'coreml' for Apple devices with Core ML support.")
        parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                            help=f"Path to the ONNX model file (default: {DEFAULT_MODEL_PATH}).")
        parser.add_argument("--labels", type=str, default=DEFAULT_LABELS_PATH,
                            help=f"Path to the labels CSV file (default: {DEFAULT_LABELS_PATH}).")
        parser.add_argument("--predictions_key", type=str, default="predictions",
                            help="Output name for label names (default: 'predictions').")
        parser.add_argument("--embeddings_key", type=str, default="embeddings",
                            help="Output name for embeddings (default: 'embeddings').")
        parser.add_argument("--label_format", type=str,  default="{sci_name}_{com_name}",
                            help="Format of the labels in the CSV file (default: '{sci_name}_{com_name}').")
        parser.add_argument("--out_postfix", type=str, default="onnx_results",
                            help="Output format for predictions (default: 'onnx_results').")
        parser.add_argument("--supported_exts", type=str, default="wav,mp3,m4a,aac,flac,ogg",
                            help="Supported file extensions for predictions (default: 'wav,mp3,m4a,aac,flac,ogg').")
        parser.add_argument("--sampling_rate", type=float, default=32000,
                            help="Sampling rate [Hz] for audio processing (default: 32000).")
        return parser

    def process_cli(self, args) -> None:
        self.configure({
            "device": args.device,
            "model": args.model,
            "labels": args.labels,
            "predictions_key": args.predictions_key,
            "embeddings_key": args.embeddings_key,
            "label_format": args.label_format,
            "out_postfix": args.out_postfix,
            "sampling_rate": args.sampling_rate,
            "chunk_length": args.chunk_length,
            "overlap": args.overlap,
            "min_conf": args.min_conf,
            "supported_exts": args.supported_exts
        })

        # is file or directory
        if os.path.isdir(args.path):
            files = [os.path.join(args.path, f) for f in os.listdir(
                args.path) if os.path.isfile(os.path.join(args.path, f)) and f.split('.')[-1] in args.supported_exts.split(',')]
            self.predict_batch(files)
        else:
            self.predict(args.path)

    @staticmethod
    def download_defaults(model_path: str, labels_path: str) -> None:
        # Download default model if missing
        if model_path == DEFAULT_MODEL_PATH and not os.path.isfile(model_path):
            logging.info(
                f"Default model not found. Downloading:\n  {DEFAULT_MODEL_URL}\n  -> {model_path}")
            if not download_file(DEFAULT_MODEL_URL, model_path):
                logging.error("Failed to download default model.")
                return False
        # Download default labels if missing
        if labels_path == DEFAULT_LABELS_PATH and not os.path.isfile(labels_path):
            logging.info(
                f"Default labels not found. Downloading:\n  {DEFAULT_LABELS_URL}\n  -> {labels_path}")
            if not download_file(DEFAULT_LABELS_URL, labels_path):
                logging.error("Failed to download default labels.")
                return False
        return True

    def format_label(self, row: dict) -> str:
        """
        Format label string based on the specified label_format.
        The format can include placeholders like {sci_name}, {com_name}, etc.
        """
        try:
            return self.label_format.format(**row)
        except KeyError as e:
            logging.error(
                f"Missing key in label formatting: {e}. Check your label_format and CSV columns.")
            return row.get("label", "unknown")

    def load_labels(self, labels_csv: str) -> List[str]:
        # CSV is semicolon-delimited with columns: id;sci_name;com_name;gbif;class;order
        labels = []
        with open(labels_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                labels.append(self.format_label(row))
        if not labels:
            raise ValueError(f"No labels found in {labels_csv}")
        return labels

    def run_onnx_inference(
            self,
        chunks: np.ndarray,
        batch_size: int = 16,
        return_embeddings: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Run inference with ONNX model.

        Args:
            chunks: [N, T] float32 mono audio.
            batch_size: batch size.
            return_embeddings: if True, also return stacked embeddings [N, D].

        Returns:
            predictions: [N, C] float32
            embeddings: [N, D] float32 or None
        """
        if chunks.shape[0] == 0:
            return np.zeros((0, 0), dtype=np.float32), None

        # Get input/output info
        input_name = self.session.get_inputs()[0].name
        input_type = self.session.get_inputs()[0].type
        output_names = [o.name for o in self.session.get_outputs()]

        # Determine input dtype (handle FP16 models)
        if "float16" in input_type:
            input_dtype = np.float16
        else:
            input_dtype = np.float32

        preds_out: List[np.ndarray] = []
        embs_out: List[np.ndarray] = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size].astype(input_dtype)
            outputs = self.session.run(output_names, {input_name: batch})

            # print(len(chunks), outputs[3].shape)

            # Model outputs: predictions, embeddings (two outputs) or just predictions
            pred_key_index = output_names.index(self.predictions_key)
            if len(outputs) == 2:
                emb_key_index = output_names.index(self.embeddings_key)

                pred = outputs[pred_key_index]
                emb = outputs[emb_key_index]
                if return_embeddings:
                    embs_out.append(emb.astype(np.float32))
            else:
                pred = outputs[pred_key_index]

            # do softmax if model output is logits (not probabilities)
            if np.any(pred < 0) or np.any(pred > 1):
                pred = np.exp(pred) / np.sum(np.exp(pred),
                                             axis=-1, keepdims=True)

            preds_out.append(pred.astype(np.float32))

        predictions = np.concatenate(preds_out, axis=0)
        embeddings = np.concatenate(
            embs_out, axis=0) if return_embeddings and embs_out else None
        return predictions, embeddings

    def chunks_to_dataframe(
        self,
        audio_path: str,
        spans: List[Tuple[float, float]],
        probs_chunks: np.ndarray,
        min_conf: float,
    ) -> pd.DataFrame:
        base = os.path.basename(audio_path)
        data = {
            "name": [],
            "start_sec": [],
            "end_sec": [],
            "confidence": [],
            "label": [],
        }
        for ci, ((start, end), probs) in enumerate(zip(spans, probs_chunks)):
            if probs.ndim != 1:
                probs = probs.ravel()
            idx = np.where(probs >= min_conf)[0]
            if idx.size == 0:
                continue
            sort_order = np.argsort(-probs[idx])
            for j in idx[sort_order]:
                conf = float(probs[j])
                data["name"].append(base)
                data["start_sec"].append(round(start, 3))
                data["end_sec"].append(round(end, 3))
                data["confidence"].append(round(conf, 6))
                data["label"].append(self.labels[j])
        return pd.DataFrame(data)

    def load_model(self):
        self.download_defaults(self.model_path, self.labels_path)

        self.labels = self.load_labels(self.labels_path)

        try:
            if self.device == "coreml":
                providers = [
                    ('CoreMLExecutionProvider', {
                        "ModelFormat": "MLProgram", "MLComputeUnits": "ALL",
                        "RequireStaticInputShapes": "0", "EnableOnSubgraphs": "0"
                    }),
                    "CPUExecutionProvider"
                ]
            elif self.device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

            session = ort.InferenceSession(
                self.model_path, providers=providers)

            self.session = session
            # Report actual provider used
            actual_provider = session.get_providers(
            )[0] if session.get_providers() else "unknown"
            logging.info(f"ONNX provider: {actual_provider}")
        except Exception as e:
            logging.error(f"Error loading ONNX model: {e}")

    def load_audio(self, file_path: str) -> tuple:
        try:
            y, sr = librosa.load(file_path, sr=self.sampling_rate, mono=True)

            return y, sr
        except Exception as e:
            error_msg = str(e)
            if any(ext in file_path.lower() for ext in [".m4a", ".mp4", ".aac"]):
                error_msg += "\nNote: Loading M4A/AAC files requires ffmpeg to be installed and available in your system PATH."
            logging.error(f"Error loading audio: {error_msg}")

    def predict_batch(self, files: List[str]) -> dict:
        predictions = {}
        for file_path in files:
            logging.info(f"Processing file: {file_path}")
            result = self.predict(file_path)
            predictions[file_path] = result
        return predictions

    def predict(
        self,
        file_path: str,
    ) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[str]]:
        if self.session is None:
            self.load_model()

        y, sr = self.load_audio(file_path)
        if y is None or sr is None:
            logging.error("Audio loading failed. Prediction aborted.")
            return None, None, None

        chunks, spans = chunk_audio(
            y, self.chunk_length, overlap=self.overlap, sr=self.sampling_rate)
        if len(chunks) == 0:
            logging.error("No audio samples to process.")
            return None, None, None

        probs_chunks, embeddings = self.run_onnx_inference(
            chunks, return_embeddings=False,
        )
        if probs_chunks.shape[-1] != len(self.labels):
            logging.error(
                f"Error: Model output shape {probs_chunks.shape[-1]} does not match labels count {len(self.labels)}.")
            logging.error(
                "This usually happens when you use a custom or filtered model but run it with the default labels file (or vice versa).")
            logging.error(
                "Please ensure that you specify the correct labels CSV file using the '--labels' argument.")
            return None, None, None

        out_path = os.path.splitext(file_path)[0] + f".{self.out_postfix}.csv"

        df = self.chunks_to_dataframe(
            file_path, spans, probs_chunks, self.min_conf)
        df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)

        logging.info(
            f"Chunks processed: {len(chunks)}; detections exported: {df.shape[0]} (min_conf={self.min_conf}, overlap={self.overlap}, export_embeddings={False})")
        logging.info(f"CSV: {out_path}")
        logging.info(f"SR={self.sampling_rate}, Device={self.device}")

        return df, embeddings, out_path


if __name__ == "__main__":
    init_default_logger()

    predictor = ONNXBioacousticsPredictor()
    parser = predictor.parse_args()
    args = parser.parse_args()
    predictor.process_cli(args)
