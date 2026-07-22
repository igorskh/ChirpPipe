import logging

from chirps.audio.normalize import NormalizeAudio
from chirps.ml.onnx_bioacoustics import ONNXBioacousticsPredictor

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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


if __name__ == "__main__":
    main()
