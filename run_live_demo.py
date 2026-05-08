#!/usr/bin/env python3
"""
Run the PiFocus local live demo with one command.
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
    parser = argparse.ArgumentParser(description="Run the PiFocus live demo.")
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
        "--camera_interval",
        type=float,
        default=1,
        help="Seconds between replayed camera prediction rows. Defaults to 1.",
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


def fusion_loop(
    manager: ProcessManager,
    stop_event: threading.Event,
    fusion_interval: float,
) -> None:
    while not stop_event.is_set():
        process = manager.popen([PYTHON, "src/fusion_engine.py"])
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

        stop_event.wait(fusion_interval)


def main() -> int:
    args = parse_args()

    if args.duration < 1:
        print("--duration must be at least 1 second", file=sys.stderr)
        return 2
    if args.fusion_interval <= 0:
        print("--fusion_interval must be greater than 0", file=sys.stderr)
        return 2
    if args.camera_interval <= 0:
        print("--camera_interval must be greater than 0", file=sys.stderr)
        return 2

    manager = ProcessManager()
    stop_event = threading.Event()
    dashboard_process: subprocess.Popen[bytes] | None = None

    def handle_stop(_signum: int, _frame: object) -> None:
        stop_event.set()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    print("Starting camera replay stream...", flush=True)
    manager.popen(
        [
            PYTHON,
            "src/replay_camera_stream.py",
            "--interval",
            str(args.camera_interval),
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

    print("Starting fusion updater...", flush=True)
    fusion_thread = threading.Thread(
        target=fusion_loop,
        args=(manager, stop_event, args.fusion_interval),
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
        print("Stopping PiFocus demo...", flush=True)
        stop_event.set()
        manager.terminate_all()
        fusion_thread.join(timeout=6)


if __name__ == "__main__":
    raise SystemExit(main())
