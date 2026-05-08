#!/usr/bin/env python3
"""
Predict visual attention states from camera feature CSV rows using a trained
scikit-learn model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
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
]
PRESERVED_COLUMNS = ["timestamp", "elapsed_seconds"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict camera-based visual attention from feature CSV rows."
    )
    parser.add_argument(
        "--input_csv",
        default="data/live_camera_features.csv",
        help="Input camera feature CSV path. Defaults to data/live_camera_features.csv.",
    )
    parser.add_argument(
        "--model_path",
        default="models/visual_attention_model.pkl",
        help="Trained scikit-learn model path. Defaults to models/visual_attention_model.pkl.",
    )
    parser.add_argument(
        "--output_csv",
        default="data/live_camera_predictions.csv",
        help="Output predictions CSV path. Defaults to data/live_camera_predictions.csv.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def load_inputs(input_csv: Path, model_path: Path) -> tuple[pd.DataFrame, object]:
    if not input_csv.exists():
        raise FileNotFoundError(f"input CSV not found: {input_csv}")
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")

    df = pd.read_csv(input_csv)
    require_columns(df, PRESERVED_COLUMNS + FEATURE_COLUMNS, "input CSV")
    if df.empty:
        raise ValueError("input CSV has no rows to predict")

    model = joblib.load(model_path)
    return df, model


def positive_class_probability(model: object, features: pd.DataFrame) -> pd.Series:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        classes = list(getattr(model, "classes_", []))

        if len(probabilities.shape) != 2 or probabilities.shape[1] == 0:
            raise ValueError("model predict_proba returned an unexpected shape")

        if 1 in classes:
            positive_index = classes.index(1)
        elif "focused" in classes:
            positive_index = classes.index("focused")
        elif "1" in classes:
            positive_index = classes.index("1")
        elif True in classes:
            positive_index = classes.index(True)
        else:
            return pd.Series(0.0, index=features.index)

        return pd.Series(probabilities[:, positive_index], index=features.index).clip(0, 1)

    raise ValueError("model must provide predict_proba for visual attention probability")


def heuristic_probability(df: pd.DataFrame) -> pd.Series:
    if "focus_score" not in df.columns:
        return pd.Series(0.0, index=df.index)

    return pd.to_numeric(df["focus_score"], errors="coerce").fillna(0).clip(0, 1)


def visual_attention_state(probability: float) -> str:
    if probability >= 0.7:
        return "visually_engaged"
    if probability >= 0.4:
        return "uncertain"
    return "not_engaged"


def predict_attention(df: pd.DataFrame, model: object) -> pd.DataFrame:
    features = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        missing_rows = int(features.isna().any(axis=1).sum())
        raise ValueError(f"input CSV has non-numeric or missing feature values in {missing_rows} rows")

    model_probabilities = positive_class_probability(model, features)
    heuristic_probabilities = heuristic_probability(df)
    probabilities = pd.Series(
        np.maximum(model_probabilities, heuristic_probabilities),
        index=df.index,
    ).clip(0, 1)

    output = df[PRESERVED_COLUMNS].copy()
    output["visual_attention_pred"] = (probabilities >= 0.55).astype(int)
    output["visual_attention_prob"] = probabilities
    output["visual_attention_state"] = output["visual_attention_prob"].apply(
        visual_attention_state
    )
    output["model_prob_focused"] = model_probabilities
    output["heuristic_prob"] = heuristic_probabilities
    return output


def print_counts(output: pd.DataFrame) -> None:
    print("prediction counts:")
    for state, count in output["visual_attention_state"].value_counts().items():
        print(f"  {state}: {count}")
    print("first 10 prediction rows:")
    print(
        output[
            [
                "elapsed_seconds",
                "model_prob_focused",
                "heuristic_prob",
                "visual_attention_prob",
                "visual_attention_state",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


def main() -> int:
    args = parse_args()
    output_csv = Path(args.output_csv)

    try:
        df, model = load_inputs(Path(args.input_csv), Path(args.model_path))
        print("model.classes_:", getattr(model, "classes_", None))
        output = predict_attention(df, model)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_csv, index=False)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote output CSV: {output_csv}")
    print_counts(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
