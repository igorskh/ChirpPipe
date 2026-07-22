"""Tool definitions for chirppipe MCP server."""

import os
from typing import Any
from mcp.types import TextContent, ImageContent

from mcp_registry import registry, ToolDefinition
from mcp_tools import (
    CropAudioTool, NormalizeAudioTool, SonogramTool, RmsBarsTool,
    MIDIMarkersTool, GlobalOccurrencesTool, BioacousticsPredictorTool,
    AudacityMarkersTool, BioClipTool, LarIqaTool
)


def _crop_audio_handler(arguments: dict[str, Any]) -> TextContent:
    tool = CropAudioTool()
    tool.configure({
        "start_time": arguments.get("start_time", 0.0),
        "end_time": arguments.get("end_time"),
        "sample_rate": arguments.get("sample_rate", 32000),
        "out_postfix": arguments.get("out_postfix", "cropped")
    })
    result = tool.process(path=arguments["path"])
    return TextContent(type="text", text=f"Cropped audio saved to: {result['output_path']}")


def _normalize_audio_handler(arguments: dict[str, Any]) -> TextContent:
    tool = NormalizeAudioTool()
    tool.configure({
        "normalize_level": arguments.get("level", -1.0),
        "sample_rate": arguments.get("sample_rate", 32000),
        "out_postfix": arguments.get("out_postfix", "normalized")
    })
    result = tool.process(path=arguments["path"])
    return TextContent(type="text", text=f"Normalized audio saved to: {result['output_path']}")


def _generate_sonogram_handler(arguments: dict[str, Any]) -> list[ImageContent]:
    tool = SonogramTool()
    tool.configure({
        "hop_length": arguments.get("hop_length", 512),
        "n_fft": arguments.get("n_fft", 2048),
        "sample_rate": arguments.get("sample_rate", 32000),
        "output_format": arguments.get("output_format", "jpg")
    })
    img_base64 = tool.process(path=arguments["path"])
    return [ImageContent(type="image", data=img_base64, mimeType="image/jpeg")]


def _compute_rms_bars_handler(arguments: dict[str, Any]) -> TextContent:
    tool = RmsBarsTool()
    tool.configure({
        "resolution": arguments.get("resolution", 0.5),
        "sample_rate": arguments.get("sample_rate", 32000)
    })
    result = tool.process(path=arguments["path"])
    return TextContent(type="text", text=f"RMS bars: {result['rms_bars']}")


def _export_midi_markers_handler(arguments: dict[str, Any]) -> TextContent:
    tool = MIDIMarkersTool()
    tool.configure({
        "csv_path": arguments["csv_path"],
        "output": arguments.get("output"),
        "min_confidence": arguments.get("min_confidence", 0.0),
        "ppqn": arguments.get("ppqn", 960),
        "tempo": arguments.get("tempo", 120.0)
    })
    tool.process(
        csv_path=arguments["csv_path"],
        output=arguments.get("output"),
        min_confidence=arguments.get("min_confidence", 0.0),
        ppqn=arguments.get("ppqn", 960),
        tempo=arguments.get("tempo", 120.0)
    )
    return TextContent(type="text", text="MIDI markers exported successfully")


def _predict_species_handler(arguments: dict[str, Any]) -> TextContent:
    tool = BioacousticsPredictorTool()
    tool.configure({
        "preset": arguments.get("preset"),
        "device": arguments.get("device", "cpu"),
        "sampling_rate": arguments.get("sampling_rate", 32000),
        "chunk_length": arguments.get("chunk_length", 3.0),
        "overlap": arguments.get("overlap", 0.0),
        "min_conf": arguments.get("min_conf", 0.2)
    })
    result = tool.process(path=arguments["path"])
    return TextContent(
        type="text",
        text=f"Predictions saved to: {result['out_path']}. Top predictions: {result['dataframe'].head().to_string()}"
    )


def _global_occurrences_handler(arguments: dict[str, Any]) -> TextContent:
    tool = GlobalOccurrencesTool()
    results = tool.process(
        scientific_name=arguments.get("scientific_name"),
        country_code=arguments.get("country_code"),
        month=arguments.get("month"),
        language=arguments.get("language_code", "en")
    )

    if results is None:
        return TextContent(
            type="text",
            text=f"No occurrences found for scientific name '{arguments.get('scientific_name')}'.")

    result_text = f"Results for scientific name '{arguments.get('scientific_name')}':\n"
    result_text += f"Localized Name: {results['localized_name']}\n"
    result_text += f"Total Occurrences: {results['total_occurrences']}\n"
    return TextContent(type="text", text=result_text)


def _audacity_markers_handler(arguments: dict[str, Any]) -> TextContent:
    tool = AudacityMarkersTool()
    tool.configure({
        "csv_path": arguments["csv_path"],
        "output": arguments.get("output"),
        "min_confidence": arguments.get("min_confidence", 0.0)
    })
    result = tool.process(
        csv_path=arguments["csv_path"],
        output=arguments.get("output"),
        min_confidence=arguments.get("min_confidence", 0.0)
    )
    return TextContent(
        type="text",
        text=f"Audacity labels exported to: {result['output_path']}. Number of labels: {result['num_labels']}"
    )


def _bioclip_inference_handler(arguments: dict[str, Any]) -> TextContent:
    tool = BioClipTool()
    tool.configure({
        "threshold": arguments.get("threshold", 0.2),
        "output_format": arguments.get("output_format", "json")
    })
    result = tool.process(
        image_path=arguments["image_path"],
        threshold=arguments.get("threshold", 0.2),
        output_format=arguments.get("output_format", "json")
    )

    result_test = ""
    for p in result["predictions"]:
        if p['score'] < arguments.get("threshold", 0.2):
            continue
        file_name = p['file_name'].split(os.sep)[-1]
        result_test += f"{file_name}: {p['common_name']} ({p['species']}) [{p['score']:.2f}]\n"

    return TextContent(type="text", text=f"BioClip inference completed.\n{result_test}")


def _lar_iqa_assess_handler(arguments: dict[str, Any]) -> TextContent:
    tool = LarIqaTool()
    result = tool.process(input=arguments["input"])

    text_result = ""
    for r in result["results"]:
        image_path = r["image_path"]
        score = r["score"]
        text_result += f"LAR-IQA score for {image_path}: {score:.4f}\n"
    return TextContent(type="text", text=text_result)


def register_tools() -> None:
    """Register all tools with the registry."""

    registry.register(ToolDefinition(
        name='crop_audio',
        description='Crop audio files to a specified time range and save as new WAV file.',
        input_schema={
            'type': 'object',
            'properties': {
                'path': {'type': 'string', 'description': 'Path to the input audio file.'},
                'start_time': {'type': 'number', 'description': 'Start time in seconds.', 'default': 0.0},
                'end_time': {'type': 'number', 'description': 'End time in seconds (None means end of audio).'},
                'sample_rate': {'type': 'number', 'description': 'Sample rate for audio processing.', 'default': 32000},
                'out_postfix': {'type': 'string', 'description': 'Output postfix for cropped audio files.', 'default': 'cropped'}
            },
            'required': ['path']
        },
        handler=_crop_audio_handler
    ))

    registry.register(ToolDefinition(
        name='normalize_audio',
        description='Normalize audio files to a specified dBFS level (Audacity-style peak normalization).',
        input_schema={
            'type': 'object',
            'properties': {
                'path': {'type': 'string', 'description': 'Path to the input audio file.'},
                'level': {'type': 'number', 'description': 'Normalization level in dBFS.', 'default': -1.0},
                'sample_rate': {'type': 'number', 'description': 'Sample rate for audio processing.', 'default': 32000},
                'out_postfix': {'type': 'string', 'description': 'Output postfix for normalized audio files.', 'default': 'normalized'}
            },
            'required': ['path']
        },
        handler=_normalize_audio_handler
    ))

    registry.register(ToolDefinition(
        name='generate_sonogram',
        description='Generate MEL sonogram image from audio files.',
        input_schema={
            'type': 'object',
            'properties': {
                'path': {'type': 'string', 'description': 'Path to the input audio file.'},
                'hop_length': {'type': 'number', 'description': 'Hop length for STFT.', 'default': 512},
                'n_fft': {'type': 'number', 'description': 'FFT size.', 'default': 2048},
                'sample_rate': {'type': 'number', 'description': 'Sample rate for audio processing.', 'default': 32000},
                'output_format': {'type': 'string', 'description': 'Output image format.', 'default': 'jpg'}
            },
            'required': ['path']
        },
        handler=_generate_sonogram_handler
    ))

    registry.register(ToolDefinition(
        name='compute_rms_bars',
        description='Compute RMS values per time bar from audio files and return as JSON.',
        input_schema={
            'type': 'object',
            'properties': {
                'path': {'type': 'string', 'description': 'Path to the input audio file.'},
                'resolution': {'type': 'number', 'description': 'Time resolution in seconds for bars.', 'default': 0.5},
                'sample_rate': {'type': 'number', 'description': 'Sample rate for audio processing.', 'default': 32000}
            },
            'required': ['path']
        },
        handler=_compute_rms_bars_handler
    ))

    registry.register(ToolDefinition(
        name='export_midi_markers',
        description='Generate a Standard MIDI File with BirdNET detections as cue markers for Logic Pro.',
        input_schema={
            'type': 'object',
            'properties': {
                'csv_path': {'type': 'string', 'description': 'Path to the BirdNET detections CSV file.'},
                'output': {'type': 'string', 'description': 'Output MIDI file path.'},
                'min_confidence': {'type': 'number', 'description': 'Minimum confidence threshold for detections.', 'default': 0.0},
                'ppqn': {'type': 'number', 'description': 'Ticks per quarter note resolution.', 'default': 960},
                'tempo': {'type': 'number', 'description': 'Tempo (BPM) to embed in the MIDI file.', 'default': 120.0}
            },
            'required': ['csv_path']
        },
        handler=_export_midi_markers_handler
    ))

    registry.register(ToolDefinition(
        name='predict_species',
        description='Run BirdNET V3 species detection on audio files using ONNX model.',
        input_schema={
            'type': 'object',
            'properties': {
                'path': {'type': 'string', 'description': 'Path to the audio file or folder to process.'},
                'preset': {'type': 'string', 'description': 'Preset configuration for model and labels.', 'enum': ['birdnet_v3', 'perch_v2']},
                'chunk_length': {'type': 'number', 'description': 'Length of each audio chunk in seconds.', 'default': 3.0},
                'overlap': {'type': 'number', 'description': 'Overlap between consecutive chunks in seconds.', 'default': 0.0},
                'min_conf': {'type': 'number', 'description': 'Minimum confidence threshold for predictions.', 'default': 0.2},
                'device': {'type': 'string', 'description': 'Device to run inference on.', 'default': 'cpu', 'enum': ['cpu', 'coreml']},
                'model': {'type': 'string', 'description': 'Path to the ONNX model file.'},
                'labels': {'type': 'string', 'description': 'Path to the labels CSV file.'},
                'sampling_rate': {'type': 'number', 'description': 'Sampling rate for audio processing.', 'default': 32000}
            },
            'required': ['path']
        },
        handler=_predict_species_handler
    ))

    registry.register(ToolDefinition(
        name='global_occurrences',
        description='Get global occurrences for a given species from the GBIF database and localized names.',
        input_schema={
            'type': 'object',
            'properties': {
                'scientific_name': {'type': 'string', 'description': 'Scientific name of the species to search for.'},
                'country_code': {'type': 'string', 'description': 'Country code to filter occurrences.'},
                'month': {'type': 'string', 'description': 'Month to filter occurrences (1-12).'},
                'language_code': {'type': 'string', 'description': 'Language code for localized name (default: \'en\').'}
            },
            'required': ['scientific_name']
        },
        handler=_global_occurrences_handler
    ))

    registry.register(ToolDefinition(
        name='audacity_markers_exporter',
        description='Generate Audacity labels from BirdNET detections CSV file.',
        input_schema={
            'type': 'object',
            'properties': {
                'csv_path': {'type': 'string', 'description': 'Path to the BirdNET detections CSV file.'},
                'output': {'type': 'string', 'description': 'Output Audacity labels file path.'},
                'min_confidence': {'type': 'number', 'description': 'Minimum confidence threshold for detections.', 'default': 0.0}
            },
            'required': ['csv_path']
        },
        handler=_audacity_markers_handler
    ))

    registry.register(ToolDefinition(
        name='bioclip_inference',
        description='Run BioClip inference using ONNX model on image or folder.',
        input_schema={
            'type': 'object',
            'properties': {
                'image_path': {'type': 'string', 'description': 'Path to the input image or directory.'},
                'threshold': {'type': 'number', 'description': 'Threshold for prediction confidence.', 'default': 0.2},
                'output_format': {'type': 'string', 'description': 'Output format for predictions.', 'default': 'json', 'enum': ['json', 'csv']}
            },
            'required': ['image_path']
        },
        handler=_bioclip_inference_handler
    ))

    registry.register(ToolDefinition(
        name='LAR-IQA_assess',
        description='Run LAR-IQA non-reference image quality assessment on image or folder.',
        input_schema={
            'type': 'object',
            'properties': {
                'input': {'type': 'string', 'description': 'Path to the input image file or directory.'}
            },
            'required': ['input']
        },
        handler=_lar_iqa_assess_handler
    ))
