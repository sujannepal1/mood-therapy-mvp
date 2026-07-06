"""
librosa_demo.py  —  PRESENTATION USE ONLY

Demonstrates the full pipeline from raw audio to song recommendation.
This is the "we built it from scratch" demo for the defense.

Flow:
  raw audio file → librosa feature extraction → MLP classifier → emotion → ISO recommender → 3 songs

Run with:
  uv run python demo/librosa_demo.py <path_to_audio_file.mp3>

Requires a local MP3 or WAV file — place any test files in demo/
(they are git-ignored via .gitignore)
"""

import sys
import numpy as np
import librosa
import joblib
import pandas as pd


MODELS_DIR = "models"
DATASET_PATH = "data/processed/dataset_labeled.csv"


def extract_features(audio_path: str) -> dict:
    """Extract audio features from a local MP3/WAV file using librosa."""
    # TODO: load audio, extract tempo, RMS energy, MFCCs, chroma, spectral centroid
    pass


def predict_emotion(features: dict) -> str:
    """Load trained MLP and predict emotion from extracted features."""
    # TODO: load mlp .pkl, align feature order, predict
    pass


def run(audio_path: str) -> None:
    # TODO: extract → predict → recommend → print
    pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python demo/librosa_demo.py <audio_file.mp3>")
        sys.exit(1)
    run(sys.argv[1])
