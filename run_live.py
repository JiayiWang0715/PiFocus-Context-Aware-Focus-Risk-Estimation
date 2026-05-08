#!/usr/bin/env python3
"""
Run the PiFocus local monitoring pipeline.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable or "python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PiFocus live monitoring pipeline.")
    parser.add_argument(
        "--task_mode",
        default="writing",
        help="Task mode passed to the activity logger. Defaults to writing.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=900,
        help="Activity logger duration in seconds. Defaults to 900.",
    )
    parser.add_argument(
        "--fusion_interval",
        type=float,
        default=3,
        help="Seconds between fusion engine runs. Defaults to 3.",
    )
    parser.add_argument(
        "--prediction_interval",
        type=float,
        default=3,
        help="Seconds between camera prediction updates. Defaults to 3.",
    )
    parser.add_argument(
        "--camera_features_csv",
        default="data/live_camera_features.csv",
        help="Live camera feature CSV written by the camera feature extractor.",
    )
    parser.add_argument(
        "--camera_predictions_csv",
        default="data/live_camera_predictions.csv",
        help="Live camera prediction CSV consumed by the fusion engine.",
    )
    parser.add_argument(
        "--start_camera",
        action="store_true",
        help="Start src/pi_camera_feature_extractor.py in this process group.",
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
        help="OpenCV camera index passed to the camera feature extractor.",
    )
    return parser.parse_args()


class ProcessManager:
    def __init__(self) -> None:
        self._processes: set[subprocess.Popen[bytes]] = set()
        self._lock = threading.Lock()

    def popen(self, command: list[str]) -> subprocess.Popen[bytes]:
        process = subprocess.Popen(command, cwd=PROJECT_ROOT)
        with self._lock:
            self._processes.add(process)
        return process

    def discard(self, process: subprocess.Popen[bytes]) -> None:
        with self._lock:
            self._processes.discard(process)

    def terminate_all(self) -> None:
        with self._lock:
            processes = list(self._processes)

        for process in processes:
            if process.poll() is None:
                process.terminate()

        deadline = time.monotonic() + 5
        for process in processes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass

        for process in processes:
            if process.poll() is None:
                process.kill()


def run_repeating_command(
    manager: ProcessManager,
    stop_event: threading.Event,
    command: list[str],
    interval: float,
) -> None:
    while not stop_event.is_set():
        process = manager.popen(command)
        try:
            while process.poll() is None and not stop_event.is_set():
                time.sleep(0.1)
        finally:
            if stop_event.is_set() and process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            manager.discard(process)

        stop_event.wait(interval)


def main() -> int:
    args = parse_args()

    if args.duration < 1:
        print("--duration must be at least 1 second", file=sys.stderr)
        return 2
    if args.fusion_interval <= 0:
        print("--fusion_interval must be greater than 0", file=sys.stderr)
        return 2
    if args.prediction_interval <= 0:
        print("--prediction_interval must be greater than 0", file=sys.stderr)
        return 2

    manager = ProcessManager()
    stop_event = threading.Event()

    def handle_stop(_signum: int, _frame: object) -> None:
        stop_event.set()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    if args.start_camera:
        print("Starting camera feature extractor...", flush=True)
        manager.popen(
            [
                PYTHON,
                "src/pi_camera_feature_extractor.py",
                "--output_csv",
                args.camera_features_csv,
                "--camera_index",
                str(args.camera_index),
            ]
        )

    print("Starting activity logger...", flush=True)
    manager.popen(
        [
            PYTHON,
            "laptop/activity_logger.py",
            "--task_mode",
            args.task_mode,
            "--duration",
            str(args.duration),
        ]
    )

    print("Starting camera prediction updater...", flush=True)
    prediction_thread = threading.Thread(
        target=run_repeating_command,
        args=(
            manager,
            stop_event,
            [
                PYTHON,
                "src/predict_camera_attention.py",
                "--input_csv",
                args.camera_features_csv,
                "--output_csv",
                args.camera_predictions_csv,
            ],
            args.prediction_interval,
        ),
        daemon=True,
    )
    prediction_thread.start()

    print("Starting fusion updater...", flush=True)
    fusion_thread = threading.Thread(
        target=run_repeating_command,
        args=(
            manager,
            stop_event,
            [
                PYTHON,
                "src/fusion_engine.py",
                "--camera_csv",
                args.camera_predictions_csv,
            ],
            args.fusion_interval,
        ),
        daemon=True,
    )
    fusion_thread.start()

    try:
        print("Starting dashboard...", flush=True)
        dashboard_process = manager.popen(["streamlit", "run", "frontend/dashboard.py"])
        return dashboard_process.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        print("Stopping PiFocus pipeline...", flush=True)
        stop_event.set()
        manager.terminate_all()
        prediction_thread.join(timeout=6)
        fusion_thread.join(timeout=6)


if __name__ == "__main__":
    raise SystemExit(main())
