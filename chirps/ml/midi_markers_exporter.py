import argparse
import csv
import struct
import sys
from collections import defaultdict

from chirps.cli_chirp import CLIChirp
from chirps.utils import init_default_logger


class MIDIMarkersExporter(CLIChirp):
    """CLI tool to embed BirdNET detections as cue markers in a WAV file."""

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="Generate a Standard MIDI File with BirdNET detections as markers, "
            "for Logic Pro's marker-import-on-open behavior."
        )
        parser.add_argument("csv_path")
        parser.add_argument("-o", "--output", default="markers.mid")
        parser.add_argument("--min-confidence", type=float, default=0.0)
        parser.add_argument("--ppqn", type=int, default=960,
                            help="Ticks per quarter note resolution (default: 960)")
        parser.add_argument("--tempo", type=float, default=120.0,
                            help="Tempo (BPM) to embed in the MIDI file (default: 120). "
                                "Keep this in mind - the same tempo must remain active in "
                                "the Logic project for marker times to stay correct.")
        return parser

    def process_cli(self, args) -> None:
        detections = MIDIMarkersExporter.load_detections(
            args.csv_path, min_confidence=args.min_confidence)
        if not detections:
            sys.exit("No detections found after filtering — nothing to write.")

        ticks_per_second = MIDIMarkersExporter.build_midi_file(
            detections, args.output,
            tempo_bpm=args.tempo, ppqn=args.ppqn,
        )

        print(f"Wrote {len(detections)} marker(s) to {args.output}")
        print(
            f"Tempo: {args.tempo} BPM, PPQN: {args.ppqn} ({ticks_per_second:.2f} ticks/sec)")
        for det in detections:
            text = MIDIMarkersExporter.build_marker_text(
                det["candidates"])
            print(f"  {det['start_sec']:.3f}s  {text}")
        print()
        print("In Logic Pro: File > Open... this .mid file (creates a new project),")
        print(
            f"then import your WAV at bar 1 / sample 0, and keep the project tempo at {args.tempo} BPM.")

    @staticmethod
    def parse_label(label):
        if "_" in label:
            scientific, common = label.split("_", 1)
        else:
            scientific, common = "", label
        return scientific.strip(), common.strip()

    @staticmethod
    def load_detections(csv_path, min_confidence=0.0):
        windows = defaultdict(list)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"name", "start_sec", "end_sec", "confidence", "label"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"CSV is missing required column(s): {', '.join(sorted(missing))}")
            for row in reader:
                confidence = float(row["confidence"])
                if confidence < min_confidence:
                    continue
                start_sec = float(row["start_sec"])
                end_sec = float(row["end_sec"])
                scientific, common = MIDIMarkersExporter.parse_label(row["label"])
                windows[(start_sec, end_sec)].append(
                    (confidence, scientific, common))

        result = []
        for (start_sec, end_sec), candidates in sorted(windows.items()):
            candidates.sort(key=lambda c: c[0], reverse=True)
            result.append(
                {"start_sec": start_sec, "end_sec": end_sec, "candidates": candidates})
        return result

    @staticmethod
    def build_marker_text(candidates):
        top_confidence, _, top_common = candidates[0]
        parts = [f"{top_common} {top_confidence * 100:.0f}%"]
        for confidence, scientific, common in candidates:
            parts.append(f"{common} {confidence * 100:.0f}%")
        return " / ".join(parts)

    @staticmethod
    def write_varlen(value):
        """Encode an integer as a MIDI variable-length quantity."""
        if value < 0:
            raise ValueError("Variable-length quantities must be non-negative")
        buf = [value & 0x7F]
        value >>= 7
        while value:
            buf.append((value & 0x7F) | 0x80)
            value >>= 7
        return bytes(reversed(buf))

    @staticmethod
    def meta_event(delta_ticks, meta_type, data):
        return (
            MIDIMarkersExporter.write_varlen(delta_ticks)
            + b"\xff"
            + bytes([meta_type])
            + MIDIMarkersExporter.write_varlen(len(data))
            + data
        )

    @staticmethod
    def build_midi_file(detections, output_path, tempo_bpm=120.0, ppqn=960):
        ticks_per_second = ppqn * (tempo_bpm / 60.0)
        microseconds_per_quarter = round(60_000_000 / tempo_bpm)

        events = []  # list of (absolute_tick, meta_type, data_bytes)

        # Set Tempo at time 0
        tempo_data = struct.pack(">I", microseconds_per_quarter)[1:]  # 3 bytes
        events.append((0, 0x51, tempo_data))

        for det in detections:
            tick = round(det["start_sec"] * ticks_per_second)
            text = MIDIMarkersExporter.build_marker_text(
                det["candidates"])
            events.append((tick, 0x06, text.encode("utf-8")))

        # Sort by absolute tick; tempo event (tick 0) naturally comes first
        # since it's added before any marker and ties are stable-sorted.
        events.sort(key=lambda e: e[0])

        track_data = bytearray()
        last_tick = 0
        for abs_tick, meta_type, data in events:
            delta = abs_tick - last_tick
            track_data += MIDIMarkersExporter.meta_event(delta, meta_type, data)
            last_tick = abs_tick

        # End of Track
        track_data += MIDIMarkersExporter.meta_event(0, 0x2F, b"")

        header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ppqn)
        track_chunk = b"MTrk" + \
            struct.pack(">I", len(track_data)) + bytes(track_data)

        with open(output_path, "wb") as f:
            f.write(header + track_chunk)

        return ticks_per_second


if __name__ == "__main__":
    init_default_logger()

    exporter = MIDIMarkersExporter()
    parser = exporter.parse_args()
    args = parser.parse_args()
    exporter.process_cli(args)
