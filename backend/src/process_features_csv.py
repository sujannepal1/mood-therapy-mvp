"""
process_features_csv.py

Converts the teammate's features.csv (raw librosa output, 300+ columns)
into our nepali_dataset format (7 features + emotion label).

Saves: data/processed/nepali_dataset_500.csv

Run:
  cd backend
  uv run python src/process_features_csv.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH  = Path("data/raw/features.csv")
OUTPUT_PATH = Path("data/processed/nepali_dataset_500.csv")

# Krumhansl-Schmuckler profiles (same as collect_spotify_data.py)
_KS_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
_KS_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

CHROMA_COLS = [
    "chroma_C_mean", "chroma_C#_mean", "chroma_D_mean", "chroma_D#_mean",
    "chroma_E_mean", "chroma_F_mean",  "chroma_F#_mean","chroma_G_mean",
    "chroma_G#_mean","chroma_A_mean",  "chroma_A#_mean","chroma_B_mean",
]

# Same emotion zones as label_emotions.py
EMOTION_CENTROIDS = {
    "Happy":    (0.80, 0.80),
    "Sad":      (0.20, 0.20),
    "Angry":    (0.20, 0.80),
    "Fear":     (0.20, 0.60),
    "Disgust":  (0.15, 0.50),
    "Surprise": (0.60, 0.85),
}


def label_emotion(v: float, e: float, m: int) -> str:
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
        return "Surprise"
    return min(
        EMOTION_CENTROIDS,
        key=lambda em: (v - EMOTION_CENTROIDS[em][0]) ** 2 + (e - EMOTION_CENTROIDS[em][1]) ** 2,
    )


def detect_mode(row: pd.Series) -> int:
    chroma = np.array([row[c] for c in CHROMA_COLS], dtype=float)
    major_corr = max(float(np.corrcoef(np.roll(_KS_MAJOR, i), chroma)[0, 1]) for i in range(12))
    minor_corr = max(float(np.corrcoef(np.roll(_KS_MINOR, i), chroma)[0, 1]) for i in range(12))
    return 1 if major_corr >= minor_corr else 0


def clean_track_name(file_name: str) -> str:
    name = re.sub(r"\.[^.]+$", "", file_name)           # remove extension
    name = re.sub(r"\[[A-Za-z0-9_\-]{8,12}\]", "", name)  # remove YouTube IDs
    name = re.sub(r"\s{2,}", " ", name).strip()
    return name


def run() -> None:
    print(f"Loading {INPUT_PATH}…")
    df = pd.read_csv(INPUT_PATH)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    rows = []
    for i, row in df.iterrows():
        # ── Mode (Krumhansl-Schmuckler on chroma means) ───────────────────────
        mode = detect_mode(row)

        # ── Energy ───────────────────────────────────────────────────────────
        energy = float(np.clip(row["rms_mean"] / 0.15, 0, 1))

        # ── Tempo ─────────────────────────────────────────────────────────────
        tempo      = float(row["tempo_bpm"])
        tempo_norm = float(np.clip((tempo - 60) / 120, 0, 1))

        # ── Loudness ──────────────────────────────────────────────────────────
        loudness = float(20 * np.log10(row["rms_mean"] + 1e-10))

        # ── Acousticness ──────────────────────────────────────────────────────
        acousticness = float(np.clip(row["harmonic_energy_ratio"], 0, 1))

        # ── Danceability (normalise beat_regularity to 0-1) ───────────────────
        danceability = float(np.clip(row["beat_regularity"] / 0.07, 0, 1))

        # ── Valence (same formula as collect_spotify_data.py) ─────────────────
        centroid_norm = float(np.clip(row["spectral_centroid_mean"] / 4000, 0, 1))
        valence = float(np.clip(0.40 * mode + 0.35 * tempo_norm + 0.25 * centroid_norm, 0.05, 0.95))

        # ── Emotion label ─────────────────────────────────────────────────────
        emotion = label_emotion(valence, energy, mode)

        rows.append({
            "track_id":    f"feat_{i:04d}",
            "track_name":  clean_track_name(str(row["file_name"])),
            "artists":     "",
            "emotion":     emotion,
            "valence":     round(valence,      3),
            "energy":      round(energy,       3),
            "tempo":       round(tempo,        1),
            "mode":        mode,
            "loudness":    round(loudness,     2),
            "acousticness":round(acousticness, 3),
            "danceability":round(danceability, 3),
        })

    out = pd.DataFrame(rows)

    print(f"\nEmotion distribution:")
    dist = out["emotion"].value_counts()
    for emotion, count in dist.items():
        bar = "█" * int(count / 3)
        print(f"  {emotion:<10} {count:>4}  {bar}")

    print(f"\nFeature ranges:")
    print(out[["valence","energy","tempo","mode","loudness","acousticness","danceability"]].describe().round(3).to_string())

    # Cap each emotion at 100 songs so one label doesn't dominate training.
    # Drop emotions with fewer than 10 songs — too few to be useful and misleading.
    kept = []
    for emotion, group in out.groupby("emotion"):
        if len(group) >= 10:
            kept.append(group.head(100))
    out = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=out.columns)

    print(f"\nAfter capping (max 100 per emotion, min 10 to include):")
    dist2 = out["emotion"].value_counts()
    for emotion, count in dist2.items():
        bar = "█" * int(count / 3)
        print(f"  {emotion:<10} {count:>4}  {bar}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved → {OUTPUT_PATH}  ({len(out)} tracks)")


if __name__ == "__main__":
    run()
