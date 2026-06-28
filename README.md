# ChirpPipe

Python utilities for bioacoustics processing.

## Modules

### BirdNet ONNX predictioms
```bash
uv run python -m chirps.ml.bird_net_v3 [file_or_folder_path]
```

### Convert CSV predictions to MIDI markers
```bash
uv run python -m chirps.ml.midi_markers_exporter [file_path]
```