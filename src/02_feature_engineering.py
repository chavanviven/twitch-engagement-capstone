"""
02_feature_engineering.py
--------------------------
Builds engineered features on top of the cleaned Twitch dataset.

Core idea: raw stats (watch time, followers, etc.) mostly just reflect
channel SIZE. Two channels can have similar follower counts but very
different audience quality. We engineer ratio-based features that
capture EFFICIENCY / ENGAGEMENT QUALITY independent of channel size,
then combine them into a single Engagement Efficiency Score (EES).

Engineered features:
  - avg_to_peak_ratio      : avg_viewers / peak_viewers
                              -> how consistent viewership is vs one-off spikes
  - watch_per_follower     : watch_time_min / followers
                              -> how much total watch-time each follower is worth
  - views_per_follower     : views_gained / followers
                              -> reach efficiency relative to existing follower base
  - follower_growth_rate   : followers_gained / followers
                              -> % growth relative to existing base (normalizes size)
  - viewers_per_stream_hr  : avg_viewers / (stream_time_min / 60)
                              -> audience density per hour streamed
  - stickiness             : watch_time_min / (stream_time_min * avg_viewers)
                              -> how much longer viewers stay engaged than
                                 the "expected" baseline watch time

All ratio features are min-max scaled to [0, 1] before combining into
the Engagement Efficiency Score (EES), an equally-weighted composite.
"""

import pandas as pd

IN_PATH = "data/processed/twitch_clean.csv"
OUT_PATH = "data/processed/twitch_features.csv"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["avg_to_peak_ratio"] = df["avg_viewers"] / df["peak_viewers"]
    df["watch_per_follower"] = df["watch_time_min"] / df["followers"]
    df["views_per_follower"] = df["views_gained"] / df["followers"]
    df["follower_growth_rate"] = df["followers_gained"] / df["followers"]
    df["viewers_per_stream_hr"] = df["avg_viewers"] / (df["stream_time_min"] / 60)
    df["stickiness"] = df["watch_time_min"] / (
        df["stream_time_min"] * df["avg_viewers"]
    )

    return df


def winsorize(series: pd.Series, lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.Series:
    """Cap extreme outliers at the given percentiles before scaling.

    A handful of channels have very few followers relative to their
    watch time / views, which produces ratio values 5-40x larger than
    the rest of the dataset. Left uncapped, these single points dominate
    min-max scaling and any distance-based clustering on top of it.
    Winsorizing keeps the ranking largely intact while preventing a
    few outliers from compressing everyone else's scaled range.
    """
    lower = series.quantile(lower_pct)
    upper = series.quantile(upper_pct)
    return series.clip(lower=lower, upper=upper)


def min_max_scale(series: pd.Series) -> pd.Series:
    return (series - series.min()) / (series.max() - series.min())


def compute_engagement_efficiency_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ratio_cols = [
        "avg_to_peak_ratio",
        "watch_per_follower",
        "views_per_follower",
        "follower_growth_rate",
        "viewers_per_stream_hr",
        "stickiness",
    ]

    scaled_cols = []
    for col in ratio_cols:
        winsorized = winsorize(df[col])
        scaled_col = f"{col}_scaled"
        df[scaled_col] = min_max_scale(winsorized)
        scaled_cols.append(scaled_col)

    # Equal-weighted composite score (justified in report: no prior basis
    # to weight one engagement dimension over another, so equal weighting
    # is the defensible default; sensitivity to weighting is discussed
    # in Results & Discussion).
    df["engagement_efficiency_score"] = df[scaled_cols].mean(axis=1)

    return df


if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)
    df = engineer_features(df)
    df = compute_engagement_efficiency_score(df)

    print("Engineered columns added:")
    new_cols = [c for c in df.columns if c not in pd.read_csv(IN_PATH).columns]
    print(new_cols)
    print("\nEES summary:")
    print(df["engagement_efficiency_score"].describe())

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved feature-engineered data -> {OUT_PATH}")
