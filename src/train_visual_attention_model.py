#!/usr/bin/env python3
"""
Train the PiFocus visual attention model from labeled camera feature rows.

This script keeps the model training path reproducible: it reads the same camera
feature schema produced by src/pi_camera_feature_extractor.py and saves the
Random Forest model used by src/predict_camera_attention.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


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
TARGET_COLUMN = "label"
LABEL_MAP = {
    "distracted": 0,
    "not_focused": 0,
    "not engaged": 0,
    "not_engaged": 0,
    "focused": 1,
    "likely_focused": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PiFocus visual attention classifiers from labeled camera features."
    )
    parser.add_argument(
        "--input_csv",
        default="data/session_features_v2.csv",
        help="Labeled camera feature CSV. Defaults to data/session_features_v2.csv.",
    )
    parser.add_argument(
        "--model_output",
        default="models/visual_attention_model.pkl",
        help="Random Forest model output path. Defaults to models/visual_attention_model.pkl.",
    )
    parser.add_argument(
        "--results_dir",
        default="results",
        help="Directory for confusion matrices and feature importance plots.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test split fraction. Defaults to 0.2.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducible training. Defaults to 42.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def load_training_data(input_csv: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not input_csv.exists():
        raise FileNotFoundError(f"training CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    require_columns(df, FEATURE_COLUMNS + [TARGET_COLUMN], str(input_csv))

    features = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    labels = df[TARGET_COLUMN].astype(str).str.strip().str.lower().map(LABEL_MAP)

    valid_rows = labels.notna() & ~features.isna().any(axis=1)
    features = features.loc[valid_rows].copy()
    labels = labels.loc[valid_rows].astype(int).copy()

    if features.empty:
        raise ValueError("training CSV has no valid labeled rows")
    if labels.nunique() < 2:
        raise ValueError("training CSV must contain at least two label classes")

    return features, labels


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str,
    output_path: Path,
) -> None:
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["distracted", "focused"],
    )
    plt.title(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def print_evaluation(name: str, y_true: pd.Series, y_pred: pd.Series) -> None:
    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["distracted", "focused"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def train(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv)
    model_output = Path(args.model_output)
    results_dir = Path(args.results_dir)

    X, y = load_training_data(input_csv)

    print("===== Label Distribution =====")
    print(y.value_counts().rename(index={0: "distracted", 1: "focused"}))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_scaled, y_train)
    log_pred = log_model.predict(X_test_scaled)
    print_evaluation("Logistic Regression", y_test, log_pred)
    save_confusion_matrix(
        y_test,
        log_pred,
        "Logistic Regression Confusion Matrix",
        results_dir / "logistic_regression_confusion_matrix.png",
    )

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=args.random_state,
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    print_evaluation("Random Forest Classifier", y_test, rf_pred)

    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_model, model_output)
    print(f"\nSaved model to {model_output}")

    save_confusion_matrix(
        y_test,
        rf_pred,
        "Random Forest Confusion Matrix",
        results_dir / "random_forest_confusion_matrix.png",
    )

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": rf_model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    print("\n===== Random Forest Feature Importance =====")
    print(importance_df.to_string(index=False))

    importance_df.plot(
        x="feature",
        y="importance",
        kind="bar",
        legend=False,
    )
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(results_dir / "random_forest_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nSaved output figures:")
    print(f"- {results_dir / 'logistic_regression_confusion_matrix.png'}")
    print(f"- {results_dir / 'random_forest_confusion_matrix.png'}")
    print(f"- {results_dir / 'random_forest_feature_importance.png'}")


def main() -> int:
    args = parse_args()
    try:
        train(args)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
