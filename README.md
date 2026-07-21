# ChirpPipe

Python utilities for nature observations processing - images and bioacoustics.

Implementation includes MCP (model context protocol) server to easily connect all modules to a chatbot.

## Modules

### BirdNet v3 ONNX inference
```bash
uv run python -m chirps.ml.onnx_bioacoustics demo/2034488.wav
```

### Perch 2 ONNX inference
```bash
uv run python -m chirps.ml.onnx_bioacoustics --model models/perch_v2.onnx --labels models/perch_2_labels.csv --chunk_length=5 --predictions_key=label  --label_format="{inat2024_fsd50k}" demo/2034488.wav
```

### BioCLIP 2 Inference
```bash
uv run python -m chirps.ml.bioclip_inference demo/P7181094.jpeg 
```

### Convert CSV predictions to MIDI markers
```bash
uv run python -m chirps.ml.midi_markers_exporter [file_path]
```

## Usage in code
Example pipiline for normalizing audio and inference:

```python
import logging

from chirps.audio.normalize import NormalizeAudio
from chirps.ml.onnx_bioacoustics import ONNXBioacousticsPredictor


def main():
    file_path = 'demo/2034488.wav'

    normalizer = NormalizeAudio()
    normalizer.configure({"normalize_level": -3.0})

    result = normalizer.process(path=file_path)

    print(result)

    onnx_bioacoustics = ONNXBioacousticsPredictor()
    onnx_bioacoustics.configure({})
    prediction = onnx_bioacoustics.process(path=result['output_path'])

    print(prediction)

```

## MCP Server

### kit Usage
Use [kit](https://github.com/mark3labs/kit) to test the server.

Create configuration file:
```bash
cp sample.kit.yml .kit.yml
```

Run kit:
```bash
kit
```

Sample prompt:
```text
classify demo/2034488.wav
```

It should return somthing like this along with reasoning.
```text       
The classification for demo/2034488.wav is Vulpes vulpes (Red Fox) with a confidence of approximately 0.42
```

Sample prompt for image identification:
```text
identify animal in demo/P7181094.jpeg
```