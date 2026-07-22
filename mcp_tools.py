"""Tool implementations for chirppipe MCP server."""

import base64
from typing import Any

from chirps.ml.onnx_bioacoustics import ONNXBioacousticsPredictor
from chirps.ml.midi_markers_exporter import MIDIMarkersExporter
from chirps.audio.sonogram import SonogramAudio
from chirps.audio.normalize import NormalizeAudio
from chirps.audio.crop import CropAudio
from chirps.audio.rms_bars import RmsBars
from chirps.ml.audacity_markers_exporter import AudacityMarkersExporter
from chirps.ml.lar_iqa_assess import LarIqaAssess
from chirps.ml.bioclip_inference import BioClipInference


class ExifTool:
    """Tool for running ExifTool on image files."""

    def configure(self, config: dict[str, Any]) -> None:
        pass

    def process(self, path: str, set_rating: int | None = None) -> dict[str, Any]:
        from exiftool import ExifToolHelper

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

        return {"metadata": metadata}


class CropAudioTool:
    """Crop audio files to a specified time range."""

    def configure(self, config: dict[str, Any]) -> None:
        self.crop = CropAudio()
        self.crop.configure({
            "start_time": config.get("start_time", 0.0),
            "end_time": config.get("end_time"),
            "sample_rate": config.get("sample_rate", 32000),
            "out_postfix": config.get("out_postfix", "cropped")
        })

    def process(self, path: str) -> dict[str, Any]:
        return self.crop.process(path=path)


class NormalizeAudioTool:
    """Normalize audio files to a specified dBFS level."""

    def configure(self, config: dict[str, Any]) -> None:
        self.normalize = NormalizeAudio()
        self.normalize.configure({
            "normalize_level": config.get("level", -1.0),
            "sample_rate": config.get("sample_rate", 32000),
            "out_postfix": config.get("out_postfix", "normalized")
        })

    def process(self, path: str) -> dict[str, Any]:
        return self.normalize.process(path=path)


class SonogramTool:
    """Generate MEL sonogram image from audio files."""

    def configure(self, config: dict[str, Any]) -> None:
        self.sonogram = SonogramAudio()
        self.sonogram.configure({
            "hop_length": config.get("hop_length", 512),
            "n_fft": config.get("n_fft", 2048),
            "sample_rate": config.get("sample_rate", 32000),
            "output_format": config.get("output_format", "jpg")
        })

    def process(self, path: str) -> str:
        result = self.sonogram.process(path=path)
        with open(result['output_path'], "rb") as img_file:
            img_data = img_file.read()
            return base64.b64encode(img_data).decode("utf-8")


class RmsBarsTool:
    """Compute RMS values per time bar from audio files."""

    def configure(self, config: dict[str, Any]) -> None:
        self.rms_bars = RmsBars()
        self.rms_bars.configure({
            "resolution": config.get("resolution", 0.5),
            "sample_rate": config.get("sample_rate", 32000)
        })

    def process(self, path: str) -> dict[str, Any]:
        return self.rms_bars.process(path=path)


class MIDIMarkersTool:
    """Generate MIDI file with BirdNET detections as cue markers."""

    def configure(self, config: dict[str, Any]) -> None:
        self.exporter = MIDIMarkersExporter()
        self.exporter.configure({
            "csv_path": config["csv_path"],
            "output": config.get("output"),
            "min_confidence": config.get("min_confidence", 0.0),
            "ppqn": config.get("ppqn", 960),
            "tempo": config.get("tempo", 120.0)
        })

    def process(self, csv_path: str, output: str | None = None,
                min_confidence: float = 0.0, ppqn: int = 960, tempo: float = 120.0) -> None:
        args = type('Args', (), {
            'csv_path': csv_path,
            'output': output,
            'min_confidence': min_confidence,
            'ppqn': ppqn,
            'tempo': tempo
        })()
        self.exporter.process_cli(args)


class GlobalOccurrencesTool:
    """Get global occurrences for a species from GBIF database."""

    def __init__(self):
        from chirps.occurrences.global_occurrences import GlobalOccurrences
        self.occurrences = GlobalOccurrences()
        self.occurrences.configure({"data_path": "data/occurrences.db"})

    def process(self, scientific_name: str, country_code: str | None = None,
                month: str | None = None, language: str = "en") -> dict[str, Any] | None:
        return self.occurrences.process({
            "scientific_name": scientific_name,
            "country_code": country_code,
            "month": month,
            "language": language
        })


class BioacousticsPredictorTool:
    """Run species detection on audio files using ONNX model."""

    def configure(self, config: dict[str, Any]) -> None:
        self.predictor = ONNXBioacousticsPredictor()
        preset = config.get("preset")

        if preset == "birdnet_v3":
            self.predictor.configure({
                "device": config.get("device", "cpu"),
                "sampling_rate": config.get("sampling_rate", 32000),
                "chunk_length": config.get("chunk_length", 3.0),
                "overlap": config.get("overlap", 0.0),
                "min_conf": config.get("min_conf", 0.2)
            })
        elif preset == "perch_v2":
            self.predictor.configure({
                "device": config.get("device", "cpu"),
                "model": "models/perch_v2.onnx",
                "labels": "models/perch_2_labels.csv",
                "sampling_rate": config.get("sampling_rate", 32000),
                "chunk_length": config.get("chunk_length", 5.0),
                "overlap": config.get("overlap", 0.0),
                "min_conf": config.get("min_conf", 0.2),
                "predictions_key": "label",
                "label_format": "{inat2024_fsd50k}"
            })

    def process(self, path: str) -> dict[str, Any]:
        return self.predictor.process(path=path)


class AudacityMarkersTool:
    """Generate Audacity labels from BirdNET detections."""

    def configure(self, config: dict[str, Any]) -> None:
        self.exporter = AudacityMarkersExporter()
        self.exporter.configure({
            "csv_path": config["csv_path"],
            "output": config.get("output"),
            "min_confidence": config.get("min_confidence", 0.0)
        })

    def process(self, csv_path: str, output: str | None = None,
                min_confidence: float = 0.0) -> dict[str, Any]:
        return self.exporter.process({
            "csv_path": csv_path,
            "output": output,
            "min_confidence": min_confidence
        })


class BioClipTool:
    """Run BioClip inference on images using ONNX model."""

    def configure(self, config: dict[str, Any]) -> None:
        self.bioclip = BioClipInference()
        self.bioclip.configure({
            "threshold": config.get("threshold", 0.2),
            "output_format": config.get("output_format", "json")
        })

    def process(self, image_path: str, threshold: float = 0.2,
                output_format: str = "json") -> dict[str, Any]:
        return self.bioclip.process(
            image_path=image_path,
            threshold=threshold,
            output_format=output_format
        )


class LarIqaTool:
    """Run LAR-IQA non-reference image quality assessment."""

    def process(self, input: str, as_rating: bool = False) -> dict[str, Any]:
        lar_iqa_assess = LarIqaAssess()
        return lar_iqa_assess.process(input=input, as_rating=as_rating)
