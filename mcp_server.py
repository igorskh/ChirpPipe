import base64

from chirps.ml.onnx_bioacoustics import ONNXBioacousticsPredictor
from chirps.ml.midi_markers_exporter import MIDIMarkersExporter
from chirps.audio.sonogram import SonogramAudio
from chirps.audio.normalize import NormalizeAudio
from chirps.audio.crop import CropAudio
from chirps.audio.rms_bars import RmsBars
from chirps.ml.audacity_markers_exporter import AudacityMarkersExporter

from chirps.ml.lar_iqa_assess import LarIqaAssess
from chirps.ml.bioclip_inference import BioClipInference

from chirps.taxonomy.ioc_multilanguage import IOCMultilanguageTaxonomy

from mcp.types import Tool, TextContent, ImageContent
from mcp.server.stdio import stdio_server
from mcp.server import Server
from typing import Any

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_chirp_tools() -> list[Tool]:
    tools = []

    crop = CropAudio()
    crop_parser = crop.parse_args()
    crop_args = [a for a in crop_parser._actions if a.dest != "help"]
    crop_params = {}
    for arg in crop_args:
        if arg.dest != "path":
            default = arg.default if arg.default is not None else None
            crop_params[arg.dest] = {"type": "string" if isinstance(
                default, str) else "number", "default": default}

    tools.append(Tool(
        name="crop_audio",
        description="Crop audio files to a specified time range and save as new WAV file.",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the input audio file."},
                "start_time": {"type": "number", "description": "Start time in seconds.", "default": 0.0},
                "end_time": {"type": "number", "description": "End time in seconds (None means end of audio)."},
                "sample_rate": {"type": "number", "description": "Sample rate for audio processing.", "default": 32000},
                "out_postfix": {"type": "string", "description": "Output postfix for cropped audio files.", "default": "cropped"}
            },
            "required": ["path"]
        }
    ))

    tools.append(Tool(
        name="normalize_audio",
        description="Normalize audio files to a specified dBFS level (Audacity-style peak normalization).",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the input audio file."},
                "level": {"type": "number", "description": "Normalization level in dBFS.", "default": -1.0},
                "sample_rate": {"type": "number", "description": "Sample rate for audio processing.", "default": 32000},
                "out_postfix": {"type": "string", "description": "Output postfix for normalized audio files.", "default": "normalized"}
            },
            "required": ["path"]
        }
    ))

    tools.append(Tool(
        name="generate_sonogram",
        description="Generate MEL sonogram image from audio files.",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the input audio file."},
                "hop_length": {"type": "number", "description": "Hop length for STFT.", "default": 512},
                "n_fft": {"type": "number", "description": "FFT size.", "default": 2048},
                "sample_rate": {"type": "number", "description": "Sample rate for audio processing.", "default": 32000},
                "output_format": {"type": "string", "description": "Output image format.", "default": "jpg"}
            },
            "required": ["path"]
        }
    ))

    tools.append(Tool(
        name="compute_rms_bars",
        description="Compute RMS values per time bar from audio files and return as JSON.",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the input audio file."},
                "resolution": {"type": "number", "description": "Time resolution in seconds for bars.", "default": 0.5},
                "sample_rate": {"type": "number", "description": "Sample rate for audio processing.", "default": 32000}
            },
            "required": ["path"]
        }
    ))

    tools.append(Tool(
        name="export_midi_markers",
        description="Generate a Standard MIDI File with BirdNET detections as cue markers for Logic Pro.",
        inputSchema={
            "type": "object",
            "properties": {
                "csv_path": {"type": "string", "description": "Path to the BirdNET detections CSV file."},
                "output": {"type": "string", "description": "Output MIDI file path."},
                "min_confidence": {"type": "number", "description": "Minimum confidence threshold for detections.", "default": 0.0},
                "ppqn": {"type": "number", "description": "Ticks per quarter note resolution.", "default": 960},
                "tempo": {"type": "number", "description": "Tempo (BPM) to embed in the MIDI file.", "default": 120.0}
            },
            "required": ["csv_path"]
        }
    ))

    tools.append(Tool(
        name="global_occurrences",
        description="Get global occurrences for a given species from the GBIF database and localized names.",
        inputSchema={
            "type": "object",
            "properties": {
                "scientific_name": {"type": "string", "description": "Scientific name of the species to search for."},
                "country_code": {"type": "string", "description": "Country code to filter occurrences."},
                "month": {"type": "string", "description": "Month to filter occurrences (1-12)."},
                "language_code": {"type": "string", "description": "Language code for localized name (default: 'en')."}
            },
            "required": ["scientific_name"]
        }
    ))

    tools.append(Tool(
        name="predict_species",
        description="Run BirdNET V3 species detection on audio files using ONNX model. For simple usage use preset and path arguments.",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the audio file or folder to process."},
                "preset": {"type": "string", "description": "Preset configuration for model and labels.", "enum": ["birdnet_v3", "perch_v2"]},
                "chunk_length": {"type": "number", "description": "Length of each audio chunk in seconds.", "default": 3.0},
                "overlap": {"type": "number", "description": "Overlap between consecutive chunks in seconds.", "default": 0.0},
                "min_conf": {"type": "number", "description": "Minimum confidence threshold for predictions.", "default": 0.2},
                "device": {"type": "string", "description": "Device to run inference on.", "default": "cpu", "enum": ["cpu", "coreml"]},
                "model": {"type": "string", "description": "Path to the ONNX model file."},
                "labels": {"type": "string", "description": "Path to the labels CSV file."},
                "sampling_rate": {"type": "number", "description": "Sampling rate for audio processing.", "default": 32000}
            },
            "required": ["path"]
        }
    ))

    tools.append(Tool(
        name="audacity_markers_exporter",
        description="Generate Audacity labels from BirdNET detections CSV file.",
        inputSchema={
            "type": "object",
            "properties": {
                "csv_path": {"type": "string", "description": "Path to the BirdNET detections CSV file."},
                "output": {"type": "string", "description": "Output Audacity labels file path."},
                "min_confidence": {"type": "number", "description": "Minimum confidence threshold for detections.", "default": 0.0}
            },
            "required": ["csv_path"]
        }
    ))

    tools.append(Tool(
        name="bioclip_inference",
        description="Run BioClip inference on images using ONNX model.",
        inputSchema={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Path to the input image or directory."},
                "threshold": {"type": "number", "description": "Threshold for prediction confidence.", "default": 0.2},
                "output_format": {"type": "string", "description": "Output format for predictions.", "default": "json", "enum": ["json", "csv"]}
            },
            "required": ["image_path"]
        }
    ))

    tools.append(Tool(
        name="LAR-IQA_assess",
        description="Run LAR-IQA non-reference image quality assessment.",
        inputSchema={
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Path to the input image file."}
            },
            "required": ["input"]
        }
    ))

    return tools


async def handle_tool_call(name: str, arguments: dict[str, Any]) -> str:
    if name == "crop_audio":
        crop = CropAudio()
        crop.configure({
            "start_time": arguments.get("start_time", 0.0),
            "end_time": arguments.get("end_time"),
            "sample_rate": arguments.get("sample_rate", 32000),
            "out_postfix": arguments.get("out_postfix", "cropped")
        })
        result = crop.process(path=arguments["path"])
        return TextContent(type="text", text=f"Cropped audio saved to: {result['output_path']}")

    elif name == "normalize_audio":
        normalize = NormalizeAudio()
        normalize.configure({
            "normalize_level": arguments.get("level", -1.0),
            "sample_rate": arguments.get("sample_rate", 32000),
            "out_postfix": arguments.get("out_postfix", "normalized")
        })
        result = normalize.process(path=arguments["path"])
        return TextContent(type="text", text=f"Normalized audio saved to: {result['output_path']}")

    elif name == "generate_sonogram":
        sonogram = SonogramAudio()
        sonogram.configure({
            "hop_length": arguments.get("hop_length", 512),
            "n_fft": arguments.get("n_fft", 2048),
            "sample_rate": arguments.get("sample_rate", 32000),
            "output_format": arguments.get("output_format", "jpg")
        })
        result = sonogram.process(path=arguments["path"])

        # get base64 image data
        with open(result['output_path'], "rb") as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode("utf-8")

        return [ImageContent(type="image", data=img_base64, mimeType="image/jpeg")]

    elif name == "compute_rms_bars":
        rms_bars = RmsBars()
        rms_bars.configure({
            "resolution": arguments.get("resolution", 0.5),
            "sample_rate": arguments.get("sample_rate", 32000)
        })
        result = rms_bars.process(path=arguments["path"])
        return TextContent(type="text", text=f"RMS bars: {result['rms_bars']}")

    elif name == "export_midi_markers":
        exporter = MIDIMarkersExporter()
        exporter.configure({
            "csv_path": arguments["csv_path"],
            "output": arguments.get("output"),
            "min_confidence": arguments.get("min_confidence", 0.0),
            "ppqn": arguments.get("ppqn", 960),
            "tempo": arguments.get("tempo", 120.0)
        })
        exporter.process_cli(type('Args', (), arguments)())
        return TextContent(type="text", text="MIDI markers exported successfully")

    elif name == "predict_species":
        predictor = ONNXBioacousticsPredictor()
        preset = arguments.get("preset")
        if preset == "birdnet_v3":
            predictor.configure({
                "device": arguments.get("device", "cpu"),
                # "model": arguments.get("model"),
                # "labels": arguments.get("labels"),
                "sampling_rate": arguments.get("sampling_rate", 32000),
                "chunk_length": arguments.get("chunk_length", 3.0),
                "overlap": arguments.get("overlap", 0.0),
                "min_conf": arguments.get("min_conf", 0.2)
            })

        elif preset == "perch_v2":
            predictor.configure({
                "device": arguments.get("device", "cpu"),
                "model": "models/perch_v2.onnx",
                "labels": "models/perch_2_labels.csv",
                "sampling_rate": arguments.get("sampling_rate", 32000),
                "chunk_length": arguments.get("chunk_length", 5.0),
                "overlap": arguments.get("overlap", 0.0),
                "min_conf": arguments.get("min_conf", 0.2),
                "predictions_key": "label",
                "label_format": "{inat2024_fsd50k}"
            })
        result = predictor.process(path=arguments["path"])
        return TextContent(type="text", text=f"Predictions saved to: {result['out_path']}. Top predictions: {result['dataframe'].head().to_string()}")

    elif name == "global_occurrences":
        from chirps.occurrences.global_occurrences import GlobalOccurrences

        occurrences = GlobalOccurrences()
        occurrences.configure({
            "data_path": "data/occurrences.db"
        })

        results = occurrences.process({
            "scientific_name": arguments.get("scientific_name"),
            "country_code": arguments.get("country_code"),
            "month": arguments.get("month"),
            "language": arguments.get("language_code", "en")
        })

        if results is None:
            return TextContent(type="text", text=f"No occurrences found for scientific name '{arguments.get('scientific_name')}'.")
        else:
            result_text = f"Results for scientific name '{arguments.get('scientific_name')}':\n"
            result_text += f"Localized Name: {results['localized_name']}\n"
            result_text += f"Total Occurrences: {results['total_occurrences']}\n"
            return TextContent(type="text", text=result_text)

    elif name == "audacity_markers_exporter":
        exporter = AudacityMarkersExporter()
        exporter.configure({
            "csv_path": arguments["csv_path"],
            "output": arguments.get("output"),
            "min_confidence": arguments.get("min_confidence", 0.0)
        })
        result = exporter.process({
            "csv_path": arguments["csv_path"],
            "output": arguments.get("output"),
            "min_confidence": arguments.get("min_confidence", 0.0)
        })
        return TextContent(type="text", text=f"Audacity labels exported to: {result['output_path']}. Number of labels: {result['num_labels']}")
    elif name == "bioclip_inference":
        bioclip = BioClipInference()
        bioclip.configure({
            "threshold": arguments.get("threshold", 0.2),
            "output_format": arguments.get("output_format", "json")
        })
        result = bioclip.process(
            image_path=arguments["image_path"],
            threshold=arguments.get("threshold", 0.2),
            output_format=arguments.get("output_format", "json")
        )
        result_test = ""
        for p in result["predictions"]:
            if p['score'] < arguments.get("threshold", 0.2):
                continue
            file_name = p['file_name'].split(os.sep)[-1]
            result_test += (
                f"{file_name}: {p['common_name']} ({p['species']}) [{p['score']:.2f}]\n")
        return TextContent(type="text", text=f"BioClip inference completed.\n{result_test}")
    elif name == "LAR-IQA_assess":
        lar_iqa_assess = LarIqaAssess()
        result = lar_iqa_assess.process(
            input=arguments["input"]
        )
        return TextContent(type="text", text=f"LAR-IQA score: {result['score']:.4f}")
    else:
        return TextContent(type="text", text=f"Unknown tool: {name}")


async def main():
    server = Server("chirppipe-mcp")
    tools = get_chirp_tools()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None = None) -> list[TextContent]:
        if arguments is None:
            arguments = {}
        result = await handle_tool_call(name, arguments)
        return result if isinstance(result, list) else [result]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
