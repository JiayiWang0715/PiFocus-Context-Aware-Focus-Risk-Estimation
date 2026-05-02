import csv
import time
from pathlib import Path
from datetime import datetime

import psutil
import win32api
import win32gui
import win32process


# --------------------------------------------------
# Configuration
# --------------------------------------------------
SAMPLE_INTERVAL_SECONDS = 2

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

LOG_FILE = DATA_DIR / "activity_log.csv"

# Change this based on the main task.
# Examples: writing, coding, reading, lecture, meeting
TASK_MODE = "writing"


# --------------------------------------------------
# App classification
# --------------------------------------------------
def classify_app(process_name, window_title):
    """
    Convert process names and window titles into user-friendly app categories.
    """

    process_name = process_name.lower()
    window_title = window_title.lower()

    if "chrome" in process_name or "msedge" in process_name:
        if "google docs" in window_title or "docs.google.com" in window_title:
            return "Google Docs"
        if "youtube" in window_title:
            return "YouTube"
        if "instagram" in window_title:
            return "Instagram"
        if "canvas" in window_title or "courseworks" in window_title:
            return "Canvas / CourseWorks"
        if "github" in window_title:
            return "GitHub"
        if "overleaf" in window_title:
            return "Overleaf"
        return "Browser"

    if "code" in process_name:
        return "VS Code"

    if "word" in process_name:
        return "Microsoft Word"

    if "powerpnt" in process_name:
        return "PowerPoint"

    if "excel" in process_name:
        return "Excel"

    if "notepad" in process_name:
        return "Notepad"

    if "discord" in process_name:
        return "Discord"

    if "wechat" in process_name:
        return "WeChat"

    if "spotify" in process_name:
        return "Spotify"

    if "explorer" in process_name:
        return "File Explorer"

    return process_name


# --------------------------------------------------
# Foreground window detection
# --------------------------------------------------
def get_foreground_window_info():
    """
    Get the currently active foreground window.
    """

    hwnd = win32gui.GetForegroundWindow()
    window_title = win32gui.GetWindowText(hwnd)

    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        process = psutil.Process(pid)
        process_name = process.name()
    except Exception:
        process_name = "unknown"

    app_name = classify_app(process_name, window_title)

    return {
        "active_app": app_name,
        "process_name": process_name,
        "window_title": window_title,
    }


def get_idle_seconds():
    """
    Estimate system idle time in seconds.
    """

    try:
        last_input_tick = win32api.GetLastInputInfo()
        current_tick = win32api.GetTickCount()
        idle_ms = current_tick - last_input_tick
        return max(0.0, idle_ms / 1000.0)
    except Exception:
        return 0.0


# --------------------------------------------------
# CSV logging
# --------------------------------------------------
def initialize_log_file():
    """
    Always create a fresh activity log file.
    """

    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "elapsed_seconds",
            "task_mode",
            "active_app",
            "window_title",
            "idle_seconds",
        ])


def append_log_row(row):
    """
    Append one activity sample to the CSV file.
    """

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# --------------------------------------------------
# Main loop
# --------------------------------------------------
def main():
    initialize_log_file()

    start_time = time.time()

    print("Activity logger started.")
    print(f"Writing samples to: {LOG_FILE}")
    print("Press Ctrl + C to stop.")

    try:
        while True:
            now = time.time()
            elapsed_seconds = now - start_time

            foreground = get_foreground_window_info()
            idle_seconds = get_idle_seconds()

            row = [
                datetime.now().isoformat(timespec="seconds"),
                round(elapsed_seconds, 2),
                TASK_MODE,
                foreground["active_app"],
                foreground["window_title"],
                round(idle_seconds, 2),
            ]

            append_log_row(row)

            time.sleep(SAMPLE_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nActivity logger stopped.")


if __name__ == "__main__":
    main()