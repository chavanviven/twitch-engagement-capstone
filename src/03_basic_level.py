"""
03_basic_level.py
------------------
BASIC / FOUNDATIONAL LEVEL PROJECT

Goal: Explore the dataset and fit a simple single-feature linear
regression to predict Average Viewers from Watch Time.

This establishes a baseline understanding of the data before moving to
the intermediate and advanced levels.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "data/processed/twitch_clean.csv"


def run_eda(df: pd.DataFrame) -> None:
    print("=== Basic Summary Statistics ===")
    print(df[["watch_time_min", "avg_viewers", "followers", "peak_viewers"]].describe())

    print("\n=== Top 5 Languages by Channel Count ===")
    print(df["language"].value_counts().head())

    print("\n=== Partnered vs Non-Partnered counts ===")
    print(df["partnered"].value_counts())

    corr = df[["watch_time_min", "stream_time_min", "peak_viewers",
               "avg_viewers", "followers", "followers_gained",
               "views_gained"]].corr()
    print("\n=== Correlation matrix ===")
    print(corr.round(2))


def plot_basic_charts(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(df["watch_time_min"], df["avg_viewers"], alpha=0.5, s=15)
    axes[0].set_xlabel("Watch Time (minutes)")
    axes[0].set_ylabel("Average Viewers")
    axes[0].set_title("Watch Time vs Average Viewers")

    df["language"].value_counts().head(8).plot(kind="bar", ax=axes[1])
    axes[1].set_title("Top 8 Languages by Channel Count")
    axes[1].set_ylabel("Number of Channels")

    plt.tight_layout()
    plt.savefig("reports/basic_eda_charts.png", dpi=150)
    print("Saved chart -> reports/basic_eda_charts.png")


def simple_regression(df: pd.DataFrame) -> None:
    X = df[["watch_time_min"]]
    y = df["avg_viewers"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\n=== Simple Linear Regression: Watch Time -> Average Viewers ===")
    print(f"Coefficient: {model.coef_[0]:.6f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"R^2 on test set: {r2_score(y_test, preds):.3f}")
    print(f"MAE on test set: {mean_absolute_error(y_test, preds):.2f}")


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    run_eda(df)
    plot_basic_charts(df)
    simple_regression(df)
