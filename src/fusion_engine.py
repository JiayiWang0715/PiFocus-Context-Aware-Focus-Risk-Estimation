#!/usr/bin/env python3
"""
Fuse camera-based attention predictions with local laptop activity context to
estimate focus risk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


CAMERA_COLUMNS = {
    "visual_attention_prob",
    "visual_attention_state",
}
ACTIVITY_COLUMNS = {
    "timestamp",
    "task_mode",
    "active_app",
    "window_title",
    "idle_seconds",
}
CAMERA_ELAPSED_COLUMNS = ("elapsed_seconds", "elapsed_seconds_x")
ACTIVITY_ELAPSED_COLUMNS = ("elapsed_seconds", "elapsed_seconds_y")

ON_TASK_KEYWORDS = {
    "coding": (
        "vs code",
        "visual studio code",
        "terminal",
        "jupyter",
        "colab",
        "github",
        "stack overflow",
        "python",
        "notebook",
    ),
    "writing": (
        "vs code",
        "visual studio code",
        "google docs",
        "microsoft word",
        "overleaf",
        "pdf",
        "preview",
        "paper",
        "latex",
        "notion",
    ),
    "reading": (
        "preview",
        "pdf",
        "chrome",
        "safari",
        "paper",
        "article",
        "arxiv",
    ),
    "lecture": (
        "zoom",
        "panopto",
        "youtube",
        "coursera",
        "canvas",
        "lecture",
        "class",
    ),
    "meeting": (
        "zoom",
        "google meet",
        "teams",
        "slack",
        "meeting",
    ),
}

OFF_TASK_KEYWORDS = (
    "youtube",
    "netflix",
    "bilibili",
    "tiktok",
    "instagram",
    "discord",
    "game",
    "entertainment",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse camera attention and laptop activity logs into focus risk."
    )
    parser.add_argument(
        "--camera_csv",
        default="data/live_camera_predictions.csv",
        help="Camera predictions CSV path. Defaults to data/live_camera_predictions.csv.",
    )
    parser.add_argument(
        "--activity_csv",
        default="data/activity_log.csv",
        help="Activity log CSV path. Defaults to data/activity_log.csv.",
    )
    parser.add_argument(
        "--output_csv",
        default="data/fused_focus_log.csv",
        help="Output fused CSV path. Defaults to data/fused_focus_log.csv.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required: Iterable[str], source: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def detect_elapsed_column(df: pd.DataFrame, candidates: Iterable[str], source: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(
        f"{source} is missing an elapsed seconds column. Expected one of: "
        + ", ".join(candidates)
    )


def read_input_csv(path: Path, required_columns: set[str], label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

    df = pd.read_csv(path)
    require_columns(df, required_columns, label)
    if df.empty:
        raise ValueError(f"{label} has no rows")
    return df


def csv_has_rows(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        return not pd.read_csv(path, nrows=1).empty
    except (pd.errors.ParserError, OSError):
        return False


def resolve_camera_csv(camera_csv: Path) -> Path:
    fallback_csv = Path("data/camera_predictions.csv")
    if csv_has_rows(camera_csv):
        return camera_csv
    if camera_csv != fallback_csv and csv_has_rows(fallback_csv):
        print(f"camera CSV has no rows; falling back to {fallback_csv}")
        return fallback_csv
    return camera_csv


def normalize_elapsed_seconds(
    df: pd.DataFrame,
    elapsed_column: str,
    source: str,
) -> pd.DataFrame:
    normalized = df.copy()
    if elapsed_column != "elapsed_seconds":
        normalized = normalized.rename(columns={elapsed_column: "elapsed_seconds"})
    extra_elapsed_columns = [
        column
        for column in {"elapsed_seconds_x", "elapsed_seconds_y"}
        if column in normalized.columns and column != elapsed_column
    ]
    if extra_elapsed_columns:
        normalized = normalized.drop(columns=extra_elapsed_columns)

    normalized["elapsed_seconds"] = pd.to_numeric(
        normalized["elapsed_seconds"], errors="coerce"
    )
    normalized = normalized.dropna(subset=["elapsed_seconds"])
    if normalized.empty:
        raise ValueError(f"{source} has no rows with valid elapsed_seconds values")
    return normalized.sort_values("elapsed_seconds").reset_index(drop=True)


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def task_relevance_score(row: pd.Series) -> float:
    task_mode = str(row.get("task_mode", "")).strip().lower()
    active_app = str(row.get("active_app", "")).strip()
    active_app_lower = active_app.lower()
    window_title = str(row.get("window_title", ""))
    combined = f"{active_app} {window_title}".lower()

    writing_off_task_apps = {
        "entertainment site",
        "youtube",
        "netflix",
        "bilibili",
        "tiktok",
        "instagram",
    }

    if task_mode == "writing" and active_app_lower in writing_off_task_apps:
        return 0.05
    if task_mode == "writing" and active_app_lower == "vs code":
        return 0.90
    if active_app_lower == "away from workspace":
        return 0.10
    if active_app_lower == "uncertain activity":
        return 0.35

    score = 0.35

    if task_mode in ON_TASK_KEYWORDS and contains_any(combined, ON_TASK_KEYWORDS[task_mode]):
        score = 0.9

    if contains_any(combined, OFF_TASK_KEYWORDS):
        if task_mode == "lecture" and "youtube" in combined:
            score = max(score, 0.75)
        else:
            score = min(score, 0.2)

    try:
        idle_seconds = float(row.get("idle_seconds", 0))
    except (TypeError, ValueError):
        idle_seconds = 0

    if idle_seconds > 600:
        score *= 0.5

    return max(0.0, min(1.0, score))


def final_state(row: pd.Series) -> str:
    visual_attention_prob = row["visual_attention_prob"]
    task_score = row["task_relevance_score"]

    if visual_attention_prob >= 0.65 and task_score >= 0.7:
        return "focused_on_task"
    if visual_attention_prob >= 0.65 and task_score < 0.5:
        return "present_but_off_task"
    if visual_attention_prob < 0.3:
        return "absent_or_disengaged"
    return "uncertain"


def add_temporal_risk_columns(fused: pd.DataFrame) -> pd.DataFrame:
    scored = fused.sort_values("elapsed_seconds").reset_index(drop=True).copy()
    scored["instantaneous_risk_score"] = 1 - (
        0.6 * scored["visual_attention_prob"] + 0.4 * scored["task_relevance_score"]
    )
    scored["instantaneous_risk_score"] = scored["instantaneous_risk_score"].clip(0, 1)

    accumulated_risk = 0.10
    consecutive_off_task_seconds = 0.0
    previous_elapsed_seconds: float | None = None
    accumulated_scores = []
    for row in scored.itertuples(index=False):
        state = row.final_state
        elapsed_seconds = float(row.elapsed_seconds)
        if previous_elapsed_seconds is None:
            elapsed_delta = 1.0
        else:
            elapsed_delta = max(0.0, min(elapsed_seconds - previous_elapsed_seconds, 5.0))
        previous_elapsed_seconds = elapsed_seconds

        if state == "focused_on_task":
            accumulated_risk -= 0.07
            consecutive_off_task_seconds = 0.0
        elif state == "present_but_off_task":
            accumulated_risk += 0.06
            consecutive_off_task_seconds += elapsed_delta
            if consecutive_off_task_seconds > 10:
                accumulated_risk += 0.03
        elif state == "absent_or_disengaged":
            accumulated_risk += 0.06
            consecutive_off_task_seconds = 0.0
        else:
            accumulated_risk += 0.015
            consecutive_off_task_seconds = 0.0

        accumulated_risk = max(0.0, min(1.0, accumulated_risk))
        accumulated_scores.append(accumulated_risk)

    scored["accumulated_risk_score"] = accumulated_scores
    scored["risk_level"] = scored["accumulated_risk_score"].apply(risk_level)
    return scored


def risk_level(score: float) -> str:
    if score < 0.35:
        return "Low Risk"
    if score < 0.65:
        return "Caution"
    return "High Risk"


def fuse(camera_csv: Path, activity_csv: Path) -> tuple[pd.DataFrame, int, int]:
    camera_csv = resolve_camera_csv(camera_csv)
    camera_df = read_input_csv(camera_csv, CAMERA_COLUMNS, "camera CSV")
    activity_df = read_input_csv(activity_csv, ACTIVITY_COLUMNS, "activity CSV")
    camera_elapsed_column = detect_elapsed_column(
        camera_df,
        CAMERA_ELAPSED_COLUMNS,
        "camera CSV",
    )
    activity_elapsed_column = detect_elapsed_column(
        activity_df,
        ACTIVITY_ELAPSED_COLUMNS,
        "activity CSV",
    )
    camera_df = normalize_elapsed_seconds(
        camera_df,
        camera_elapsed_column,
        "camera CSV",
    )
    activity_df = normalize_elapsed_seconds(
        activity_df,
        activity_elapsed_column,
        "activity CSV",
    )

    camera_df["visual_attention_prob"] = pd.to_numeric(
        camera_df["visual_attention_prob"], errors="coerce"
    ).clip(0, 1)
    camera_df = camera_df.dropna(subset=["visual_attention_prob"])
    if camera_df.empty:
        raise ValueError("camera CSV has no valid visual_attention_prob values")

    activity_df["idle_seconds"] = pd.to_numeric(
        activity_df["idle_seconds"], errors="coerce"
    ).fillna(0)

    default_task_mode = "writing"
    valid_task_modes = activity_df["task_mode"].dropna().astype(str).str.strip()
    if not valid_task_modes.empty and valid_task_modes.iloc[0]:
        default_task_mode = valid_task_modes.iloc[0]

    activity_df = activity_df.rename(columns={"timestamp": "activity_timestamp"})
    camera_df["elapsed_seconds"] = camera_df["elapsed_seconds"].astype(float)
    activity_df["elapsed_seconds"] = activity_df["elapsed_seconds"].astype(float)

    fused = pd.merge_asof(
        camera_df.sort_values("elapsed_seconds"),
        activity_df.sort_values("elapsed_seconds"),
        on="elapsed_seconds",
        direction="nearest",
        tolerance=3,
    )

    fused["active_app"] = fused["active_app"].fillna("Unknown")
    fused["window_title"] = fused["window_title"].fillna("")
    fused["task_mode"] = fused["task_mode"].fillna(default_task_mode)
    fused["idle_seconds"] = fused["idle_seconds"].fillna(0)
    if "timestamp" not in fused.columns:
        fused["timestamp"] = fused.get("activity_timestamp", "")

    fused["task_relevance_score"] = fused.apply(task_relevance_score, axis=1)
    fused["risk_score"] = 1 - (
        0.6 * fused["visual_attention_prob"] + 0.4 * fused["task_relevance_score"]
    )
    fused["risk_score"] = fused["risk_score"].clip(0, 1)
    fused["smoothed_risk_score"] = (
        fused["risk_score"].rolling(window=5, min_periods=1).mean()
    )
    fused["final_state"] = fused.apply(final_state, axis=1)
    fused = add_temporal_risk_columns(fused)

    return fused, len(camera_df), len(activity_df)


def print_summary(fused: pd.DataFrame, camera_rows: int, activity_rows: int) -> None:
    print(f"camera rows: {camera_rows}")
    print(f"activity rows: {activity_rows}")
    print(f"fused rows: {len(fused)}")
    print("active_app value counts:")
    for app, count in fused["active_app"].value_counts(dropna=False).items():
        print(f"  {app}: {count}")
    print("final_state counts:")
    for state, count in fused["final_state"].value_counts().items():
        print(f"  {state}: {count}")
    print("risk_level counts:")
    for level, count in fused["risk_level"].value_counts().items():
        print(f"  {level}: {count}")
    print("first 10 risk rows:")
    print(
        fused[
            [
                "elapsed_seconds",
                "visual_attention_prob",
                "task_relevance_score",
                "final_state",
                "accumulated_risk_score",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


def main() -> int:
    args = parse_args()
    output_csv = Path(args.output_csv)

    try:
        fused, camera_rows, activity_rows = fuse(Path(args.camera_csv), Path(args.activity_csv))
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fused.to_csv(output_csv, index=False)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_summary(fused, camera_rows, activity_rows)
    print(f"wrote output CSV: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
