import logging


def init_default_logger():
    """Initialize the default logger for the chirps package."""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
