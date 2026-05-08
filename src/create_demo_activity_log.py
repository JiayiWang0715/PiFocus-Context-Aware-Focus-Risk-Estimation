#!/usr/bin/env python3
"""
Create a privacy-safe controlled activity log for the 3-minute PiFocus demo.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path


OUTPUT_CSV = Path("data/demo_activity_log.csv")
COLUMNS = [
    "timestamp",
    "elapsed_seconds",
    "task_mode",
    "active_app",
    "window_title",
    "idle_seconds",
]


def activity_for_second(elapsed_seconds: int) -> tuple[str, str, int]:
    # These labels represent inferred activity context based on observable signals.
    # They do not imply direct detection of external devices such as phones.
    if elapsed_seconds < 30:
        return "VS Code", "PiFocus Project - Writing", 0
    if elapsed_seconds < 60:
        return "Away from Workspace", "No face detected / user away", 30
    if elapsed_seconds < 90:
        return "VS Code", "PiFocus Project - Writing", 0
    if elapsed_seconds < 120:
        return "Uncertain Activity", "Low engagement / unclear task context", 15
    if elapsed_seconds < 150:
        return "VS Code", "PiFocus Project - Writing", 0
    return "Entertainment Site", "Video content", 0


def build_rows() -> list[dict[str, object]]:
    start_time = datetime.now().replace(microsecond=0)
    rows = []
    for elapsed_seconds in range(181):
        active_app, window_title, idle_seconds = activity_for_second(elapsed_seconds)
        rows.append(
            {
                "timestamp": (start_time + timedelta(seconds=elapsed_seconds)).isoformat(),
                "elapsed_seconds": elapsed_seconds,
                "task_mode": "writing",
                "active_app": active_app,
                "window_title": window_title,
                "idle_seconds": idle_seconds,
            }
        )
    return rows


def main() -> int:
    rows = build_rows()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote output CSV: {OUTPUT_CSV}")
    print(f"rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
