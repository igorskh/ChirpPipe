
import csv
import logging
import os

from typing import List, Tuple, Optional

import librosa
import onnxruntime as ort
import numpy as np
import argparse

from chirps.cli_chirp import CLIChirp
from chirps.utils import init_default_logger

from .utils import download_file


SR = 32000  # model expects 32 kHz

DEFAULT_MODEL_PATH = "models/birdnet_v3.onnx"
DEFAULT_LABELS_PATH = "models/birdnet_v3_labels.csv"
DEFAULT_MODEL_URL = "https://zenodo.org/records/20703646/files/BirdNET+_V3.0-preview3.1_Global_11K_FP32.onnx?download=1"
DEFAULT_LABELS_URL = "https://zenodo.org/records/20703646/files/BirdNET+_V3.0-preview3.1_Global_11K_Labels.csv?download=1"


class ONNXBioacousticPredictor(CLIChirp):
    labels = []
    device = "cpu"  # default device
    session: "ort.InferenceSession" = None  # ONNX Runtime session
    predictions_key = "predictions"
    embeddings_key = "embeddings"
    label_format = "{sci_name}_{com_name}"

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
        parser.add_argument("--out_csv", type=str, default=None,
                            help="Output CSV file path (default: <input_file>.results.csv).")
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
        return parser

    def process_cli(self, args) -> None:
        chunk_length = args.chunk_length
        overlap = args.overlap
        min_conf = args.min_conf
        out_csv = args.out_csv

        self.device = args.device

        self.model_path = args.model
        self.labels_path = args.labels

        self.predictions_key = args.predictions_key
        self.embeddings_key = args.embeddings_key

        self.label_format = args.label_format

        self.predict(args.path, chunk_length=chunk_length,
                     overlap=overlap, min_conf=min_conf, out_csv=out_csv)

    @staticmethod
    def chunk_audio(y: np.ndarray, chunk_length: float, overlap: float = 0.0, sr: int = SR) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Split audio into chunks with optional temporal overlap.

        Args:
            y: 1D numpy array (mono audio).
            chunk_length: Length of each chunk in seconds (>0).
            overlap: Overlap between consecutive chunks in seconds (0 <= overlap < chunk_length).
            sr: Sample rate.

        Returns:
            chunks: Float32 array of shape [N, chunk_samples]
            spans: List of (start_sec, end_sec) for each chunk (end_sec truncated to original audio length).
        """
        chunk_len = int(round(chunk_length * sr))
        if chunk_len <= 0:
            raise ValueError("chunk_length must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= chunk_length:
            raise ValueError("overlap must be < chunk_length")

        step = chunk_len - int(round(overlap * sr))
        if step <= 0:
            raise ValueError(
                "Invalid step size (adjust overlap/chunk_length).")

        n = len(y)
        if n == 0:
            return np.zeros((0, chunk_len), dtype=np.float32), []

        starts = np.arange(0, n, step)
        chunks = []
        spans = []
        for s in starts:
            e = min(s + chunk_len, n)
            seg = y[s:e]
            if len(seg) < chunk_len:
                pad = np.zeros(chunk_len - len(seg), dtype=seg.dtype)
                seg = np.concatenate([seg, pad], axis=0)
            chunks.append(seg.astype(np.float32, copy=False))
            spans.append((s / sr, min(e, n) / sr))
            if e >= n:
                break
        return np.stack(chunks, axis=0), spans

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
                pred = np.exp(pred) / np.sum(np.exp(pred), axis=-1, keepdims=True)

            preds_out.append(pred.astype(np.float32))

        predictions = np.concatenate(preds_out, axis=0)
        embeddings = np.concatenate(
            embs_out, axis=0) if return_embeddings and embs_out else None
        return predictions, embeddings

    def save_per_chunk_csv(
        self,
        audio_path: str,
        spans: List[Tuple[float, float]],
        probs_chunks: np.ndarray,
        out_csv: str,
        min_conf: float,
        export_embeddings: bool = False,
        embeddings: Optional[np.ndarray] = None,
    ):
        """Save rows for every (chunk,label) with confidence >= min_conf, sorted by descending confidence per chunk.
        Columns:
        - name,start_sec,end_sec,confidence,label
        - if export_embeddings=True: add a single "embeddings" column containing the whole embedding vector
            serialized as a comma-separated string wrapped in quotes (handled by csv writer).
        """
        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        base = os.path.basename(audio_path)
        rows = 0

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            # default delimiter=",", quotechar='"', QUOTE_MINIMAL
            w = csv.writer(f)
            header = ["name", "start_sec", "end_sec", "confidence", "label"]
            if export_embeddings:
                # single column with the full vector as a quoted string
                header += ["embeddings"]
            w.writerow(header)

            for ci, ((start, end), probs) in enumerate(zip(spans, probs_chunks)):
                if probs.ndim != 1:
                    probs = probs.ravel()
                idx = np.where(probs >= min_conf)[0]
                if idx.size == 0:
                    continue
                sort_order = np.argsort(-probs[idx])
                # Prepare embedding string once per chunk if requested
                emb_str = None
                if export_embeddings and embeddings is not None and ci < len(embeddings):
                    vec = embeddings[ci].ravel().astype(np.float32)
                    # Comma-separated to force quoting in CSV
                    emb_str = ",".join(f"{v}" for v in vec)

                for j in idx[sort_order]:
                    conf = float(probs[j])
                    row = [base, round(start, 3), round(end, 3),
                           round(conf, 6), self.labels[j]]
                    if export_embeddings:
                        row.append("" if emb_str is None else emb_str)
                    w.writerow(row)
                    rows += 1
        return rows

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
            y, sr = librosa.load(file_path, sr=SR, mono=True)

            return y, sr
        except Exception as e:
            error_msg = str(e)
            if any(ext in file_path.lower() for ext in [".m4a", ".mp4", ".aac"]):
                error_msg += "\nNote: Loading M4A/AAC files requires ffmpeg to be installed and available in your system PATH."
            logging.error(f"Error loading audio: {error_msg}")

    def predict(
        self,
        file_path: str,
        chunk_length=3.0,
        overlap=0.0,
        min_conf=0.2,
        out_csv: Optional[str] = None,
    ) -> dict:
        if self.session is None:
            self.load_model()

        y, sr = self.load_audio(file_path)
        if y is None or sr is None:
            logging.error("Audio loading failed. Prediction aborted.")
            return {}

        chunks, spans = ONNXBioacousticPredictor.chunk_audio(
            y, chunk_length, overlap=overlap, sr=SR)
        if len(chunks) == 0:
            logging.error("No audio samples to process.")
            return {}

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
            return {}

        out_csv = out_csv if out_csv else (
            os.path.splitext(file_path)[0] + ".results.csv")
        rows = self.save_per_chunk_csv(
            file_path,
            spans,
            probs_chunks,
            out_csv,
            min_conf,
            export_embeddings=False,
            embeddings=embeddings,
        )

        logging.info(
            f"Chunks processed: {len(chunks)}; detections exported: {rows} (min_conf={min_conf}, overlap={overlap}, export_embeddings={False})")
        logging.info(f"CSV: {out_csv}")
        logging.info(f"SR={SR}, Device={self.device}")
        predictions = {}  # Replace with actual predictions

        return predictions


if __name__ == "__main__":
    init_default_logger()

    predictor = ONNXBioacousticPredictor()
    parser = predictor.parse_args()
    args = parser.parse_args()
    predictor.process_cli(args)
