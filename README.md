# Twitch Channel Analytics — Capstone Project

**DATA110 Capstone | Individual Project**

Analysis of the Top 1000 Twitch Streamers dataset across three levels of
increasing complexity, culminating in a novel engagement-efficiency
segmentation and growth-prediction study.

## Dataset

Top 1000 Twitch channels with metrics on watch time, stream time, peak
and average viewers, followers, followers gained, views gained,
partnership status, mature-content flag, and primary language.

- `data/raw/twitch_raw.csv` — original, unmodified dataset
- `data/processed/twitch_clean.csv` — cleaned dataset (see `src/01_data_cleaning.py`)
- `data/processed/twitch_features.csv` — with engineered engagement features
- `data/processed/twitch_clustered.csv` — with cluster labels from the advanced-level analysis

## Project Structure

```
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned + engineered data
├── src/
│   ├── 01_data_cleaning.py       # Data quality checks, column standardization
│   ├── 02_feature_engineering.py # Engagement Efficiency Score construction
│   ├── 03_basic_level.py         # Basic: EDA + single-feature regression
│   ├── 04_intermediate_level.py  # Intermediate: multi-feature classification
│   └── 05_advanced_level.py      # Advanced: clustering + model comparison
├── reports/                  # Generated charts and result tables
├── docs/                     # Thesis report
├── presentation/             # Viva slides
└── requirements.txt
```

## Project Levels

### 1. Basic Level — Exploratory Analysis
Summary statistics, correlation analysis, and a single-feature linear
regression predicting Average Viewers from Watch Time.
`R² = 0.406` on held-out test data.

### 2. Intermediate Level — Classification
Multi-feature Logistic Regression predicting the `Mature` content flag
from channel statistics. `Partnered` was considered first but rejected
as a target due to severe class imbalance (978/22); `Mature` (770/230)
was used instead — a deliberate, documented data-driven decision.

### 3. Advanced Level — Engagement Efficiency Segmentation & Growth Prediction
**Novel contribution:** rather than predicting raw follower counts from
raw size metrics (the common approach on public notebooks for this
dataset, which mostly re-learns "bigger channels have bigger numbers"),
this project engineers six size-independent engagement ratios (e.g.
watch-time-per-follower, viewer retention, growth rate) into a
composite **Engagement Efficiency Score**, then:

- **Segments** channels via K-Means (k=2, selected by silhouette score)
  into "High-Efficiency" vs. "Standard" engagement tiers
- **Predicts** Followers Gained using the engineered features across
  three models (Linear Regression, Random Forest, Gradient Boosting),
  with Gradient Boosting performing best (`R² = 0.337`)
- **Interprets** results: high-efficiency (niche) channels do not
  necessarily gain the most absolute followers — raw growth is still
  partly a function of channel size, not efficiency alone

## Reproducing the Analysis

```bash
pip install -r requirements.txt
python src/01_data_cleaning.py
python src/02_feature_engineering.py
python src/03_basic_level.py
python src/04_intermediate_level.py
python src/05_advanced_level.py
```

## Author

Individual capstone project — DATA110.
