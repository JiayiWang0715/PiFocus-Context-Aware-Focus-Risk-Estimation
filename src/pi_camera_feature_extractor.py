#!/usr/bin/env python3
"""
Extract live Raspberry Pi camera features for PiFocus.

The output schema matches src/predict_camera_attention.py, so the generated
feature CSV can be converted directly into visual attention predictions.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


OUTPUT_COLUMNS = [
    "timestamp",
    "elapsed_seconds",
    "avg_brightness",
    "avg_blur",
    "face_ratio",
    "missing_face_count",
    "avg_face_area_ratio",
    "avg_center_offset",
    "avg_centeredness",
    "avg_movement",
    "movement_stability",
    "focus_score",
    "attention_state",
    "label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract live camera features for PiFocus visual attention prediction."
    )
    parser.add_argument(
        "--output_csv",
        default="data/live_camera_features.csv",
        help="Output feature CSV. Defaults to data/live_camera_features.csv.",
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
        help="OpenCV camera index. Defaults to 0.",
    )
    parser.add_argument(
        "--sample_interval",
        type=float,
        default=1.0,
        help="Seconds between output rows. Defaults to 1.0.",
    )
    parser.add_argument(
        "--window_seconds",
        type=float,
        default=5.0,
        help="Rolling window used for feature aggregation. Defaults to 5.0.",
    )
    parser.add_argument(
        "--label",
        default="unlabeled",
        help="Optional label stored for later training datasets. Defaults to unlabeled.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show a local preview window on the Raspberry Pi.",
    )
    return parser.parse_args()


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def attention_state(focus_score: float, face_ratio: float) -> str:
    if face_ratio < 0.3 or focus_score < 0.4:
        return "not_engaged"
    if focus_score >= 0.7:
        return "likely_focused"
    return "uncertain_present"


def compute_focus_score(metrics: dict[str, float]) -> float:
    face_score = clamp(metrics["face_ratio"])
    centered_score = clamp(metrics["avg_centeredness"])
    missing_face_score = 1.0 - clamp(metrics["missing_face_count"] / 10.0)
    blur_score = clamp(metrics["avg_blur"] / 180.0)
    face_area_score = clamp(metrics["avg_face_area_ratio"] / 0.25)
    movement_score = clamp(metrics["movement_stability"])

    return clamp(
        0.26 * face_score
        + 0.20 * centered_score
        + 0.16 * missing_face_score
        + 0.12 * blur_score
        + 0.12 * face_area_score
        + 0.14 * movement_score
    )


def frame_metrics(
    frame: np.ndarray,
    face_detector: cv2.CascadeClassifier,
    previous_gray: np.ndarray | None,
) -> tuple[dict[str, float], np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    brightness = float(np.mean(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        face_ratio = 1.0
        face_area_ratio = float((w * h) / (width * height))
        face_center_x = x + w / 2
        face_center_y = y + h / 2
        center_offset = float(
            np.hypot(
                (face_center_x - width / 2) / (width / 2),
                (face_center_y - height / 2) / (height / 2),
            )
        )
        center_offset = clamp(center_offset)
    else:
        face_ratio = 0.0
        face_area_ratio = 0.0
        center_offset = 1.0

    centeredness = 1.0 - center_offset

    if previous_gray is None:
        movement = 0.0
    else:
        movement = float(np.mean(cv2.absdiff(gray, previous_gray)))

    movement_stability = 1.0 - clamp(movement / 40.0)

    return (
        {
            "brightness": brightness,
            "blur": blur,
            "face_ratio": face_ratio,
            "missing_face": 0.0 if face_ratio else 1.0,
            "face_area_ratio": face_area_ratio,
            "center_offset": center_offset,
            "centeredness": centeredness,
            "movement": movement,
            "movement_stability": movement_stability,
        },
        gray,
    )


def aggregate(window: deque[dict[str, float]]) -> dict[str, float]:
    if not window:
        raise ValueError("cannot aggregate an empty feature window")

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in window]))

    metrics = {
        "avg_brightness": mean("brightness"),
        "avg_blur": mean("blur"),
        "face_ratio": mean("face_ratio"),
        "missing_face_count": float(np.sum([row["missing_face"] for row in window])),
        "avg_face_area_ratio": mean("face_area_ratio"),
        "avg_center_offset": mean("center_offset"),
        "avg_centeredness": mean("centeredness"),
        "avg_movement": mean("movement"),
        "movement_stability": mean("movement_stability"),
    }
    metrics["focus_score"] = compute_focus_score(metrics)
    return metrics


def open_writer(output_csv: Path) -> tuple[object, csv.DictWriter]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_file = output_csv.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
    writer.writeheader()
    csv_file.flush()
    return csv_file, writer


def main() -> int:
    args = parse_args()
    if args.sample_interval <= 0:
        print("--sample_interval must be greater than 0", file=sys.stderr)
        return 2
    if args.window_seconds <= 0:
        print("--window_seconds must be greater than 0", file=sys.stderr)
        return 2

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_detector.empty():
        print("Error: could not load OpenCV face detector", file=sys.stderr)
        return 1

    camera = cv2.VideoCapture(args.camera_index)
    if not camera.isOpened():
        print(f"Error: could not open camera index {args.camera_index}", file=sys.stderr)
        return 1

    output_csv = Path(args.output_csv)
    csv_file, writer = open_writer(output_csv)
    feature_window: deque[dict[str, float]] = deque()
    previous_gray: np.ndarray | None = None
    start_time = time.time()
    last_write_time = 0.0

    print(f"Writing live camera features to {output_csv}", flush=True)

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Warning: camera frame read failed", file=sys.stderr)
                time.sleep(0.1)
                continue

            metrics, previous_gray = frame_metrics(frame, face_detector, previous_gray)
            now = time.time()
            feature_window.append(metrics)
            max_window_rows = max(1, int(args.window_seconds * 30))
            while len(feature_window) > max_window_rows:
                feature_window.popleft()

            if now - last_write_time >= args.sample_interval:
                elapsed_seconds = now - start_time
                output_metrics = aggregate(feature_window)
                state = attention_state(
                    output_metrics["focus_score"],
                    output_metrics["face_ratio"],
                )
                row = {
                    "timestamp": round(now, 4),
                    "elapsed_seconds": round(elapsed_seconds, 2),
                    "avg_brightness": round(output_metrics["avg_brightness"], 4),
                    "avg_blur": round(output_metrics["avg_blur"], 4),
                    "face_ratio": round(output_metrics["face_ratio"], 4),
                    "missing_face_count": round(output_metrics["missing_face_count"], 4),
                    "avg_face_area_ratio": round(output_metrics["avg_face_area_ratio"], 4),
                    "avg_center_offset": round(output_metrics["avg_center_offset"], 4),
                    "avg_centeredness": round(output_metrics["avg_centeredness"], 4),
                    "avg_movement": round(output_metrics["avg_movement"], 4),
                    "movement_stability": round(output_metrics["movement_stability"], 4),
                    "focus_score": round(output_metrics["focus_score"], 4),
                    "attention_state": state,
                    "label": args.label,
                }
                writer.writerow(row)
                csv_file.flush()
                print(
                    f"camera row {row['elapsed_seconds']}: {state} "
                    f"focus_score={row['focus_score']}",
                    flush=True,
                )
                last_write_time = now

            if args.display:
                cv2.imshow("PiFocus camera", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nStopped camera feature extraction cleanly.", file=sys.stderr)
    finally:
        camera.release()
        csv_file.close()
        if args.display:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
