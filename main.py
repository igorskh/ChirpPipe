import logging

from chirps.ml.bird_net_v3 import BirdNetV3Predictor

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    # read arbitary kwargs from command line


    model = BirdNetV3Predictor()
    model.process_cli("demo/260303_0824.wav")
    # model.load_model()

    # model.predict("demo/260303_0824.wav", chunk_length=5.0, overlap=0.5)


if __name__ == "__main__":
    main()
