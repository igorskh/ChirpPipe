# ChirpPipe

Python utilities for bioacoustics processing.

## Modules

### BirdNet v3 ONNX inference
```bash
uv run python -m chirps.ml.onnx_bioacoustic [file_or_folder_path]
```

### Perch 2 ONNX inference
```bash
uv run python -m chirps.ml.onnx_bioacoustic --model models/perch_v2.onnx --labels models/perch_2_labels.csv --chunk_length=5 --predictions_key=label  --label_format="{inat2024_fsd50k}" [file_or_folder_path]
```

### Convert CSV predictions to MIDI markers
```bash
uv run python -m chirps.ml.midi_markers_exporter [file_path]
```