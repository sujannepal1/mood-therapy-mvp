"""
label_emotions.py

Loads the raw Kaggle dataset, deduplicates by track_id,
applies rule-based emotion labels using valence + energy + mode,
and writes dataset_labeled.csv to data/processed/.

Academic basis: Russell's Circumplex Model of Affect (1980).
6 target labels: Happy, Sad, Angry, Fear, Disgust, Surprise.

Label strategy:
  - Clearly-defined zones get a rule-based label (Happy, Sad, Angry, Fear, Disgust, Surprise).
  - Surprise is given a real zone (high energy + mid-high valence) instead of being a catch-all.
  - Mid-zone songs that don't fit any rule are assigned to the nearest emotion centroid
    in (valence, energy) space — so no song is ever discarded.
"""

import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/dataset.csv")
PROCESSED_PATH = Path("data/processed/dataset_labeled.csv")

# Centroid of each emotion in Russell's Circumplex space (valence, energy)
EMOTION_CENTROIDS: dict[str, tuple[float, float]] = {
    "Happy":    (0.80, 0.80),
    "Sad":      (0.20, 0.20),
    "Angry":    (0.20, 0.80),
    "Fear":     (0.20, 0.60),
    "Disgust":  (0.15, 0.50),
    "Surprise": (0.60, 0.85),
}


def label_emotion(row: pd.Series) -> str:
    v = row["valence"]
    e = row["energy"]
    m = row["mode"]  # 1 = major key, 0 = minor key

    # Clearly-defined zones — priority ordered, no overlaps
    if v > 0.6 and e > 0.6:
        return "Happy"
    if v < 0.4 and e < 0.4:
        return "Sad"
    if v < 0.4 and e > 0.65:
        return "Angry"
    if v < 0.5 and 0.35 <= e <= 0.7 and m == 0:
        return "Fear"
    if v < 0.35 and e < 0.5:
        return "Disgust"
    if v >= 0.5 and e >= 0.65:
        return "Surprise"   # upbeat + high-energy: the genuine Surprise zone

    # Mid-zone songs that don't match any rule → nearest centroid
    return min(
        EMOTION_CENTROIDS,
        key=lambda em: (v - EMOTION_CENTROIDS[em][0]) ** 2 + (e - EMOTION_CENTROIDS[em][1]) ** 2,
    )


def run() -> None:
    print("Loading dataset...")
    df = pd.read_csv(RAW_PATH, index_col=0)
    print(f"  Loaded {len(df):,} rows")

    required = ["track_id", "valence", "energy", "mode", "track_name",
                "artists", "tempo", "danceability", "loudness",
                "acousticness", "track_genre"]
    before = len(df)
    df = df.dropna(subset=required)
    if len(df) < before:
        print(f"  Dropped {before - len(df):,} rows with missing values")

    df = df.drop_duplicates(subset="track_id", keep="first")
    print(f"  {len(df):,} unique tracks after deduplication")

    print("Labeling emotions...")
    df["emotion"] = df.apply(label_emotion, axis=1)

    distribution = df["emotion"].value_counts()
    print("\n  Emotion distribution:")
    for emotion, count in distribution.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {emotion:<10} {count:>6,}  ({pct:.1f}%)  {bar}")

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"\n  Saved → {PROCESSED_PATH}  ({len(df):,} songs)")


if __name__ == "__main__":
    run()
