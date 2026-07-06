"""
classification/classifier.py

Classifies an audio file into one of 6 emotions using the trained Nepali MLP.

Uses the same 7-feature extraction pipeline as the dataset builder —
the same librosa code that generated the training data is applied at inference,
so there's no train/inference mismatch.

Returns a dict of emotion → probability (all 6, summing to 1.0).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.features import extract_features

MLP_FEATURE_ORDER = [
    "valence", "energy", "tempo", "mode",
    "danceability", "loudness", "acousticness",
]


def classify_audio(audio_path: str, mlp_model) -> dict[str, float] | None:
    """
    Extract features from audio_path and return emotion probabilities.

    Returns None if feature extraction fails (corrupt file, unsupported format, etc.).
    """
    features = extract_features(audio_path)
    if features is None:
        return None

    x = np.array([[features[f] for f in MLP_FEATURE_ORDER]])

    probs   = mlp_model.predict_proba(x)[0]
    classes = mlp_model.classes_

    return {str(cls): round(float(prob), 4) for cls, prob in zip(classes, probs)}
