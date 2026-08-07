"""
01_data_cleaning.py
--------------------
Loads the raw Twitch dataset, checks data quality, standardizes column
names, and saves a cleaned version to data/processed/.

Dataset: Top 1000 Twitch Streamers (Watch time, Stream time, Peak/Average
viewers, Followers, Followers gained, Views gained, Partnered, Mature,
Language) — one row per channel.
"""

import pandas as pd

RAW_PATH = "data/raw/twitch_raw.csv"
OUT_PATH = "data/processed/twitch_clean.csv"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Channel": "channel",
        "Watch time(Minutes)": "watch_time_min",
        "Stream time(minutes)": "stream_time_min",
        "Peak viewers": "peak_viewers",
        "Average viewers": "avg_viewers",
        "Followers": "followers",
        "Followers gained": "followers_gained",
        "Views gained": "views_gained",
        "Partnered": "partnered",
        "Mature": "mature",
        "Language": "language",
    }
    df = df.rename(columns=rename_map)
    return df


def quality_report(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nDuplicate channel names:", df["channel"].duplicated().sum())
    print("\nNegative / zero checks (should all be False):")
    numeric_cols = [
        "watch_time_min", "stream_time_min", "peak_viewers",
        "avg_viewers", "followers", "views_gained",
    ]
    for col in numeric_cols:
        print(f"  {col} has values <= 0:", (df[col] <= 0).any())


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)

    # Drop exact duplicate rows if any slipped in
    before = len(df)
    df = df.drop_duplicates(subset="channel", keep="first")
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} duplicate channel rows.")

    # followers_gained can legitimately be negative (net loss over the
    # tracked period) -> keep as is, this is real signal, not an error.

    # Basic sanity filter: stream_time_min must be > 0 to compute
    # per-minute ratios later without divide-by-zero issues.
    df = df[df["stream_time_min"] > 0].reset_index(drop=True)

    return df


if __name__ == "__main__":
    raw = load_raw()
    quality_report(standardize_columns(raw.copy()))
    cleaned = clean(raw)
    cleaned.to_csv(OUT_PATH, index=False)
    print(f"\nSaved cleaned data -> {OUT_PATH}  (rows: {len(cleaned)})")
