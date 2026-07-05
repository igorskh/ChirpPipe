import os
import glob
import re
import logging
import argparse

from chirps.chirp_node import ChirpNode
from chirps.cli_chirp import CLIChirp


def get_file_timestamp(file_path):
    stats = os.stat(file_path)
    return stats.st_mtime, stats.st_atime


class SyncDatetime(CLIChirp, ChirpNode):
    def process(self, input_data: dict) -> dict:
        source_dir = input_data.get("source_dir")
        target_dir = input_data.get("target_dir")

        if not source_dir or not target_dir:
            logging.error("Source and target directories must be provided.")
            return {}

        if not os.path.isdir(source_dir):
            logging.error(f"Source directory does not exist: {source_dir}")
            return {}

        if not os.path.isdir(target_dir):
            logging.error(f"Target directory does not exist: {target_dir}")
            return {}

        self.sync_file_dates(source_dir, target_dir)
        return {}

    def configure(self, input_config: dict):
        pass

    def parse_args(self) -> None:
        parser = argparse.ArgumentParser(
            description="Sync file datetime based on source and target directories")

        parser.add_argument("-s", "--source_dir", type=str, required=True,
                            help="Path to the source directory containing files to sync")
        parser.add_argument("-t", "--target_dir", type=str, required=True,
                            help="Path to the target directory where files will be synced")

        return parser

    def process_cli(self, args) -> None:
        source_dir = args.source_dir
        target_dir = args.target_dir

        return self.process({
            "source_dir": source_dir,
            "target_dir": target_dir
        })

    @staticmethod
    def extract_time_offset_from_filename(filename):
        # time offset in format "1m30s" or "30s" or "1m" anywhere in the filename *+1m30s.wav
        pattern = re.compile(r"\+(\d{1,2}m)?(\d{1,2}s)?", re.IGNORECASE)
        match = pattern.search(filename)
        if match:
            minutes = int(match.group(1)[:-1]) if match.group(1) else 0
            seconds = int(match.group(2)[:-1]) if match.group(2) else 0
            return minutes * 60 + seconds
        return None

    @staticmethod
    def sync_file_datetime(source_file, target_file):
        if os.path.abspath(target_file) == os.path.abspath(source_file):
            return

        time_offset = SyncDatetime.extract_time_offset_from_filename(
            target_file)
        if time_offset is not None:
            mtime += time_offset
            atime += time_offset

        logging.info(
            f"Updating timestamp: {os.path.basename(target_file)} with {time_offset} seconds offset -> matching {os.path.basename(source_file)}"
        )

        os.utime(target_file, (atime, mtime))

    @staticmethod
    def search_for_matching_files(source_file, target_directory, source_pattern):
        filename = os.path.basename(source_file)
        match = source_pattern.match(filename)

        if match:
            prefix = match.group(1)

            mtime, atime = get_file_timestamp(source_file)

            target_search = os.path.join(target_directory, f"{prefix}_*")
            targets = [
                path for path in glob.glob(target_search)
                if path.lower().endswith(".wav")
            ]

            for target in targets:
                SyncDatetime.sync_file_datetime(source_file, target)

    @staticmethod
    def sync_file_dates(source_directory, target_directory):
        search_pattern = os.path.join(source_directory, "*")
        all_files = [
            path for path in glob.glob(search_pattern)
            if path.lower().endswith(".wav")
        ]

        logging.info(f"Found {len(all_files)} files in source directory.")

        source_pattern = re.compile(r"^(\d{6}_\d{4})\.wav$", re.IGNORECASE)

        for file_path in all_files:
            filename = os.path.basename(file_path)
            match = source_pattern.match(filename)

            if match:
                prefix = match.group(1)

                mtime, atime = get_file_timestamp(file_path)

                target_search = os.path.join(
                    target_directory, f"{prefix}_*")
                targets = [
                    path for path in glob.glob(target_search)
                    if path.lower().endswith(".wav")
                ]

                for target in targets:
                    if os.path.abspath(target) == os.path.abspath(file_path):
                        continue

                    time_offset = SyncDatetime.extract_time_offset_from_filename(
                        target)
                    if time_offset is not None:
                        mtime += time_offset
                        atime += time_offset

                    logging.info(
                        f"Updating timestamp: {os.path.basename(target)} with {time_offset} seconds offset -> matching {filename}"
                    )

                    os.utime(target, (atime, mtime))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sync_datetime = SyncDatetime()
    parser = sync_datetime.parse_args()
    args = parser.parse_args()
    sync_datetime.process_cli(args)
