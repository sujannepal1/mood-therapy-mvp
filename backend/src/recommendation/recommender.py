"""
recommendation/recommender.py

ISO Principle recommendation engine.

Given a detected emotion, calculates 3 waypoints through Russell's Circumplex
Model from the user's current emotional state toward the target (Happy).
Finds the nearest song in the Nepali dataset to each waypoint using
Euclidean distance in (valence, energy) space.

Each recommended song is classified in real-time by the trained Nepali MLP —
the emotion label on every card is a live prediction, not a CSV tag.
"""

import numpy as np
import pandas as pd
from loguru import logger

TARGET_VALENCE = 0.80
TARGET_ENERGY = 0.80

EMOTION_COORDS: dict[str, tuple[float, float]] = {
    "Happy": (0.80, 0.80),
    "Surprise": (0.60, 0.85),
    "Angry": (0.20, 0.80),
    "Fear": (0.20, 0.60),
    "Disgust": (0.15, 0.50),
    "Sad": (0.20, 0.20),
}

# 7 features — no genre (Nepali model doesn't use genre)
MLP_FEATURES = [
    "valence",
    "energy",
    "tempo",
    "mode",
    "danceability",
    "loudness",
    "acousticness",
]

TOP_K_MATCH = 20
TOP_K_BRIDGE = 15
TOP_K_DEST = 10


def spotify_url(track_id: str) -> str:
    return f"https://open.spotify.com/track/{track_id}"


def _mlp_predict(row: pd.Series, mlp_model) -> str:
    """Run a single song's audio features through the Nepali MLP."""
    try:
        features = np.array([[row[f] for f in MLP_FEATURES]])
        return str(mlp_model.predict(features)[0])
    except Exception:
        return str(row["emotion"])


def find_nearest_song(
    df: pd.DataFrame,
    v_target: float,
    e_target: float,
    exclude_ids: list[str],
    top_k: int = 20,
    mlp_model=None,
    expected_emotion: str | None = None,
) -> dict:
    candidates = df[~df["track_id"].isin(exclude_ids)].copy()
    candidates["dist"] = np.sqrt(
        (candidates["valence"] - v_target) ** 2 + (candidates["energy"] - e_target) ** 2
    )
    pool = candidates.nsmallest(top_k, "dist")

    MAX_ATTEMPTS = 3
    tried_ids: set[str] = set()
    row = pool.sample(1).iloc[0]
    emotion = (
        _mlp_predict(row, mlp_model) if mlp_model is not None else str(row["emotion"])
    )

    for attempt in range(1, MAX_ATTEMPTS + 1):
        tried_ids.add(row["track_id"])

        if expected_emotion is None or emotion == expected_emotion:
            break

        remaining = pool[~pool["track_id"].isin(tried_ids)]
        if remaining.empty or attempt == MAX_ATTEMPTS:
            logger.warning(
                f"[MLP MISMATCH] after {attempt} attempt(s): "
                f"'{row['track_name']}' predicted as {emotion!r}, "
                f"expected {expected_emotion!r} — keeping best available"
            )
            break

        logger.debug(
            f"[MLP RETRY {attempt}] '{row['track_name']}' → {emotion!r} "
            f"(expected {expected_emotion!r}), trying next"
        )
        row = remaining.sample(1).iloc[0]
        emotion = (
            _mlp_predict(row, mlp_model)
            if mlp_model is not None
            else str(row["emotion"])
        )

    logger.info(
        f"  ↳ '{row['track_name']}' by {str(row['artists'])[:40]} | "
        f"dist={float(row['dist']):.3f} | MLP→{emotion}"
    )

    return {
        "track_name": row["track_name"],
        "artists": row["artists"],
        "emotion": emotion,
        "valence": round(float(row["valence"]), 3),
        "energy": round(float(row["energy"]), 3),
        "track_id": row["track_id"],
        "spotify_url": spotify_url(row["track_id"]),
    }


def recommend(
    emotion: str,
    df: pd.DataFrame,
    mlp_model=None,
) -> list[dict]:
    if emotion not in EMOTION_COORDS:
        emotion = "Sad"

    cv, ce = EMOTION_COORDS[emotion]

    waypoints = [
        (cv, ce, TOP_K_MATCH, emotion),
        ((cv + TARGET_VALENCE) / 2, (ce + TARGET_ENERGY) / 2, TOP_K_BRIDGE, None),
        (TARGET_VALENCE, TARGET_ENERGY, TOP_K_DEST, "Happy"),
    ]

    step_labels = ["match", "bridge", "destination"]
    songs: list[dict] = []
    used_ids: list[str] = []

    for i, (v, e, k, expected) in enumerate(waypoints):
        logger.info(
            f"[WAYPOINT {i + 1}/3] {step_labels[i].upper()} | "
            f"target=({v:.2f}, {e:.2f}) | top_k={k} | expect={expected or 'any'}"
        )
        song = find_nearest_song(
            df, v, e, used_ids, top_k=k, mlp_model=mlp_model, expected_emotion=expected
        )
        songs.append(song)
        used_ids.append(song["track_id"])

    return songs
