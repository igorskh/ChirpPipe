import os
import glob
import re
import logging


def get_file_timestamp(file_path):
    stats = os.stat(file_path)
    return stats.st_mtime, stats.st_atime


class SyncDateTime:
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

        time_offset = SyncDateTime.extract_time_offset_from_filename(
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

            target_search = os.path.join(target_directory, f"{prefix}_*.wav")
            targets = glob.glob(target_search)

            for target in targets:
                SyncDateTime.sync_file_datetime(source_file, target)

    @staticmethod
    def sync_file_dates(source_directory, target_directory):
        search_pattern = os.path.join(source_directory, "*.wav")
        all_files = glob.glob(search_pattern)

        logging.info(f"Found {len(all_files)} files in source directory.")

        source_pattern = re.compile(r"^(\d{6}_\d{4})\.wav$", re.IGNORECASE)

        for file_path in all_files:
            filename = os.path.basename(file_path)
            match = source_pattern.match(filename)

            if match:
                prefix = match.group(1)

                mtime, atime = get_file_timestamp(file_path)

                target_search = os.path.join(
                    target_directory, f"{prefix}_*.wav")
                targets = glob.glob(target_search)

                for target in targets:
                    if os.path.abspath(target) == os.path.abspath(file_path):
                        continue

                    time_offset = SyncDateTime.extract_time_offset_from_filename(
                        target)
                    if time_offset is not None:
                        mtime += time_offset
                        atime += time_offset

                    logging.info(
                        f"Updating timestamp: {os.path.basename(target)} with {time_offset} seconds offset -> matching {filename}"
                    )

                    os.utime(target, (atime, mtime))
