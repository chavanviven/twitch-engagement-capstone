"""
05_advanced_level.py
----------------------
ADVANCED LEVEL / PRIMARY RESEARCH PROJECT

"Engagement Efficiency Segmentation and Growth Prediction for Twitch
Channels"

Two-part pipeline:
  Part A - Unsupervised: K-Means clustering on engineered engagement
           ratios (not raw size metrics) to segment channels into
           engagement-quality tiers, independent of channel size.
  Part B - Supervised: Compare Linear Regression, Random Forest, and
           Gradient Boosting to predict Followers Gained using the
           engineered features, then interpret feature importance.

This is the novel contribution: most public analyses of this dataset
predict raw follower counts from raw size features (watch time, peak
viewers), which mostly just re-learns "bigger channels have bigger
numbers." Here we predict growth from ENGINEERED EFFICIENCY features,
which is a genuinely different and harder question: what behavioral
/engagement patterns -- not just size -- drive follower growth?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "data/processed/twitch_features.csv"

ENGINEERED_FEATURES = [
    "avg_to_peak_ratio", "watch_per_follower", "views_per_follower",
    "follower_growth_rate", "viewers_per_stream_hr", "stickiness",
]
# Pre-scaled (winsorized + min-max scaled) versions from feature
# engineering step -- used for clustering so extreme outliers don't
# dominate the distance metric.
CLUSTER_FEATURES = [f"{c}_scaled" for c in ENGINEERED_FEATURES]
TARGET = "followers_gained"


# ---------- PART A: CLUSTERING ----------

def find_best_k(X_scaled, k_range=range(2, 7)):
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores[k] = score
        print(f"  k={k}: silhouette score = {score:.3f}")
    best_k = max(scores, key=scores.get)
    print(f"Best k by silhouette score: {best_k}")
    return best_k


def run_clustering(df: pd.DataFrame):
    # Features are already winsorized + min-max scaled in feature
    # engineering (see CLUSTER_FEATURES); re-applying StandardScaler on
    # top just standardizes them to comparable variance for K-Means.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[CLUSTER_FEATURES])

    print("=== Selecting number of clusters ===")
    best_k = find_best_k(X_scaled)

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)

    print(f"\n=== Cluster profile (mean engineered features, k={best_k}) ===")
    profile = df.groupby("cluster")[ENGINEERED_FEATURES + ["engagement_efficiency_score"]].mean()
    print(profile.round(3))

    print("\n=== Cluster sizes ===")
    print(df["cluster"].value_counts().sort_index())

    # Simple 2D visualization using top 2 features by variance for interpretability
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        df["watch_per_follower"], df["follower_growth_rate"],
        c=df["cluster"], cmap="viridis", alpha=0.6, s=20
    )
    ax.set_xlabel("Watch Time per Follower")
    ax.set_ylabel("Follower Growth Rate")
    ax.set_title(f"Channel Clusters (k={best_k}) by Engagement Pattern")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig("reports/advanced_clusters.png", dpi=150)
    print("Saved chart -> reports/advanced_clusters.png")

    return df, best_k


# ---------- PART B: SUPERVISED MODEL COMPARISON ----------

def compare_models(df: pd.DataFrame):
    X = df[ENGINEERED_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    results = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        results.append({"model": name, "r2": r2, "mae": mae})
        fitted[name] = model
        print(f"{name:20s}  R^2={r2:.3f}  MAE={mae:,.0f}")

    results_df = pd.DataFrame(results).sort_values("r2", ascending=False)
    print("\n=== Model comparison (sorted by R^2) ===")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["model"]
    best_model = fitted[best_model_name]
    print(f"\nBest model: {best_model_name}")

    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(
            best_model.feature_importances_, index=ENGINEERED_FEATURES
        ).sort_values(ascending=False)
        print("\nFeature importances (best model):")
        print(importances.round(3))

        fig, ax = plt.subplots(figsize=(7, 5))
        importances.plot(kind="barh", ax=ax)
        ax.set_title(f"Feature Importance - {best_model_name}")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig("reports/advanced_feature_importance.png", dpi=150)
        print("Saved chart -> reports/advanced_feature_importance.png")

    return results_df, fitted


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    print("########## PART A: CLUSTERING ##########\n")
    df, best_k = run_clustering(df)
    df.to_csv("data/processed/twitch_clustered.csv", index=False)
    print("Saved -> data/processed/twitch_clustered.csv")

    print("\n########## PART B: MODEL COMPARISON ##########\n")
    results_df, fitted_models = compare_models(df)
    results_df.to_csv("reports/model_comparison_results.csv", index=False)
    print("Saved -> reports/model_comparison_results.csv")
