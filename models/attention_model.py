import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Load CSV
df = pd.read_csv("session_features_v2(1).csv")

# 2. Use only required input features
features = [
    "avg_brightness",
    "avg_blur",
    "face_ratio",
    "missing_face_count",
    "avg_face_area_ratio",
    "avg_center_offset",
    "avg_centeredness",
    "avg_movement",
    "movement_stability",
    "focus_score"
]

target = "label"

X = df[features]
y = df[target]

# 3. Convert labels to numbers
y = y.map({
    "distracted": 0,
    "focused": 1
})

# 4. Check labels
print("===== Label Distribution =====")
print(y.value_counts())

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

log_pred = log_model.predict(X_test_scaled)

print("\n===== Logistic Regression =====")
print("Accuracy:", accuracy_score(y_test, log_pred))
print("\nClassification Report:")
print(classification_report(y_test, log_pred, target_names=["distracted", "focused"]))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, log_pred))

# Logistic Regression confusion matrix image
ConfusionMatrixDisplay.from_predictions(
    y_test,
    log_pred,
    display_labels=["distracted", "focused"]
)
plt.title("Logistic Regression Confusion Matrix")
plt.savefig("logistic_regression_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# 7. Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

joblib.dump(rf_model, "visual_attention_model.pkl")
print("\nSaved model to visual_attention_model.pkl")

print("\n===== Random Forest Classifier =====")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("\nClassification Report:")
print(classification_report(y_test, rf_pred, target_names=["distracted", "focused"]))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

# Random Forest confusion matrix image
ConfusionMatrixDisplay.from_predictions(
    y_test,
    rf_pred,
    display_labels=["distracted", "focused"]
)
plt.title("Random Forest Confusion Matrix")
plt.savefig("random_forest_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# 8. Feature importance for Random Forest
importance_df = pd.DataFrame({
    "feature": features,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n===== Random Forest Feature Importance =====")
print(importance_df)

# Feature importance image
importance_df.plot(
    x="feature",
    y="importance",
    kind="bar",
    legend=False
)

plt.title("Feature Importance (Random Forest)")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("random_forest_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved output figures:")
print("- logistic_regression_confusion_matrix.png")
print("- random_forest_confusion_matrix.png")
print("- random_forest_feature_importance.png")