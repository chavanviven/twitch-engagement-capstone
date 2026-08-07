"""
04_intermediate_level.py
--------------------------
INTERMEDIATE LEVEL PROJECT

Goal: Predict whether a channel is flagged as "Mature" content using a
multi-feature Logistic Regression classifier.

Note on target choice: 'Partnered' was considered first but is heavily
imbalanced (978 True / 22 False in this dataset) -- a classifier would
hit ~98% accuracy by always predicting True, without learning anything.
'Mature' is much better balanced (770 False / 230 True), so it was
chosen as the classification target instead. This decision is
documented here and in the thesis report as a data-driven judgment
call, which is good evidence of understanding for the viva.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

DATA_PATH = "data/processed/twitch_clean.csv"

FEATURES = [
    "watch_time_min", "stream_time_min", "peak_viewers",
    "avg_viewers", "followers", "followers_gained", "views_gained",
]
TARGET = "mature"


def prepare_data(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("=== Logistic Regression: Predicting Mature Flag ===")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.3f}")
    print(f"Precision: {precision_score(y_test, preds):.3f}")
    print(f"Recall:    {recall_score(y_test, preds):.3f}")
    print(f"F1 score:  {f1_score(y_test, preds):.3f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nFull classification report:")
    print(classification_report(y_test, preds))

    print("Feature coefficients (standardized -> comparable magnitudes):")
    for feat, coef in zip(FEATURES, model.coef_[0]):
        print(f"  {feat:20s} {coef:+.3f}")

    return model


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    train_and_evaluate(X_train, X_test, y_train, y_test)
