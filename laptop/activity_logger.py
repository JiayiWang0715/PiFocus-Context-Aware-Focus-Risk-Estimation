#!/usr/bin/env python3
"""
Consent-based, privacy-preserving local activity logger for macOS.

This records only active app/window metadata and idle time. It does not record
content, screenshots, or keystrokes, and it does not upload anything.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict


TASK_MODES = ("writing", "coding", "reading", "lecture", "meeting")
CSV_COLUMNS = (
    "timestamp",
    "elapsed_seconds",
    "task_mode",
    "active_app",
    "window_title",
    "idle_seconds",
)

ACTIVE_WINDOW_APPLESCRIPT = r'''
tell application "System Events"
    set frontAppProcess to first application process whose frontmost is true
    set frontAppName to name of frontAppProcess
    set windowTitle to ""
    try
        set windowTitle to name of front window of frontAppProcess
    end try
end tell
return frontAppName & "\n" & windowTitle
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log local macOS active app/window metadata and idle time to CSV."
    )
    parser.add_argument(
        "--task_mode",
        required=True,
        choices=TASK_MODES,
        help="Current task mode to attach to each logged row.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="How long to log, in seconds. Defaults to 300.",
    )
    parser.add_argument(
        "--output",
        default="data/activity_log.csv",
        help="Output CSV path. Defaults to data/activity_log.csv.",
    )
    return parser.parse_args()


def run_command(command: list[str], timeout: float = 2.0) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.stdout.strip()


def get_active_window(last_active_app: str, last_window_title: str) -> tuple[str, str]:
    try:
        output = run_command(["osascript", "-e", ACTIVE_WINDOW_APPLESCRIPT])
    except FileNotFoundError:
        return last_active_app, last_window_title
    except subprocess.TimeoutExpired:
        return last_active_app, last_window_title
    except subprocess.CalledProcessError:
        return last_active_app, last_window_title
    except OSError:
        return last_active_app, last_window_title

    lines = output.splitlines()
    active_app = lines[0] if lines else ""
    window_title = lines[1] if len(lines) > 1 else ""

    if not active_app:
        return last_active_app, last_window_title

    return active_app, window_title


def get_idle_seconds() -> int:
    command = (
        "ioreg -c IOHIDSystem | "
        "awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'"
    )
    try:
        output = run_command(["sh", "-c", command])
    except FileNotFoundError:
        return -1
    except subprocess.TimeoutExpired:
        return -1
    except subprocess.CalledProcessError:
        return -1

    try:
        return int(output)
    except ValueError:
        return -1


def ensure_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def build_row(
    task_mode: str,
    elapsed_seconds: int,
    last_active_app: str,
    last_window_title: str,
) -> tuple[Dict[str, object], str, str]:
    active_app, window_title = get_active_window(last_active_app, last_window_title)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed_seconds,
        "task_mode": task_mode,
        "active_app": active_app,
        "window_title": window_title,
        "idle_seconds": get_idle_seconds(),
    }
    return row, active_app, window_title


def print_row(row: Dict[str, object]) -> None:
    print(
        " | ".join(f"{column}={row[column]!r}" for column in CSV_COLUMNS),
        flush=True,
    )


def main() -> int:
    args = parse_args()

    if args.duration < 1:
        print("--duration must be at least 1 second", file=sys.stderr)
        return 2

    output_path = Path(args.output)
    ensure_csv(output_path)

    start = time.monotonic()
    next_log_at = start
    last_active_app = ""
    last_window_title = ""

    try:
        with output_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)

            while True:
                now = time.monotonic()
                elapsed_seconds = int(now - start)
                if elapsed_seconds >= args.duration:
                    break

                if now < next_log_at:
                    time.sleep(next_log_at - now)
                    continue

                row, last_active_app, last_window_title = build_row(
                    args.task_mode,
                    elapsed_seconds,
                    last_active_app,
                    last_window_title,
                )
                writer.writerow(row)
                csv_file.flush()
                print_row(row)

                next_log_at += 1.0
    except KeyboardInterrupt:
        print("\nStopped cleanly on Ctrl+C.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
