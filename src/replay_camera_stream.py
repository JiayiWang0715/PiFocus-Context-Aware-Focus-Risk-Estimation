#!/usr/bin/env python3
"""
Replay pre-collected camera predictions into a live-style CSV stream.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


OUTPUT_COLUMNS = [
    "elapsed_seconds",
    "visual_attention_pred",
    "visual_attention_prob",
    "visual_attention_state",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay pre-collected Raspberry Pi camera predictions."
    )
    parser.add_argument(
        "--input_csv",
        default="data/camera_predictions.csv",
        help="Input camera predictions CSV. Defaults to data/camera_predictions.csv.",
    )
    parser.add_argument(
        "--output_csv",
        default="data/live_camera_predictions.csv",
        help="Output live-style camera predictions CSV. Defaults to data/live_camera_predictions.csv.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between replayed rows. Defaults to 1.0.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop back to the first row when the replay reaches the end.",
    )
    return parser.parse_args()


def read_rows(input_csv: Path) -> list[dict[str, str]]:
    if not input_csv.exists():
        raise FileNotFoundError(f"input CSV not found: {input_csv}")

    with input_csv.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = [column for column in OUTPUT_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"{input_csv} is missing required columns: {', '.join(missing)}"
            )
        rows = [{column: row.get(column, "") for column in OUTPUT_COLUMNS} for row in reader]

    if not rows:
        raise ValueError(f"{input_csv} has no rows")
    return rows


def write_header(output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()


def append_row(output_csv: Path, row: dict[str, str]) -> None:
    with output_csv.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    if args.interval <= 0:
        print("--interval must be greater than 0", file=sys.stderr)
        return 2

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    try:
        rows = read_rows(input_csv)
        write_header(output_csv)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # This script replays pre-collected Raspberry Pi camera predictions for a stable demo. It does not fabricate predictions.
    index = 0
    try:
        while index < len(rows):
            row = rows[index]
            append_row(output_csv, row)
            print(
                f"replayed row {index}: {row.get('visual_attention_state', 'unknown')}",
                flush=True,
            )
            index += 1
            if index >= len(rows) and args.loop:
                index = 0
            if index < len(rows) or args.loop:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped camera replay cleanly.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
