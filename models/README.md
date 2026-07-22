# Prepare models

## BirdNet V3
Download ONNX model and labels from Zenodo:

- **Model**: [BirdNET V3.0 Global 11K FP32 ONNX](https://zenodo.org/records/20703646/files/BirdNET+_V3.0-preview3.1_Global_11K_FP32.onnx?download=1)
  ```bash
  wget "https://zenodo.org/records/20703646/files/BirdNET+_V3.0-preview3.1_Global_11K_FP32.onnx?download=1" -O models/birdnet_v3.onnx
  ```

- **Labels**: [BirdNET V3.0 Labels CSV](https://zenodo.org/records/20703646/files/BirdNET+_V3.0-preview3.1_Global_11K_Labels.csv?download=1)
  ```bash
  wget "https://zenodo.org/records/20703646/files/BirdNET+_V3.0-preview3.1_Global_11K_Labels.csv?download=1" -O models/birdnet_v3_labels.csv
  ```

## Perch V2
Download ONNX model and labels:

- **Model**: Download Perch V2 ONNX from [TensorFlow Hub](https://tfhub.dev/google/bird-vocalization-classifier/perch/2) or [Hugging Face](https://huggingface.co/google/perch)
  - Save as `models/perch_v2.onnx`

- **Labels**: Download the corresponding labels file
  - Save as `models/perch_2_labels.csv`
  - Format: CSV with inat2024_fsd50k label mappings


## LAR-IQA
Download models as instructed in [LAR-IQA README](https://github.com/nasimjamshidi/LAR-IQA#installation) and place `.pt` files in this folder.