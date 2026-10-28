"""
pipeline.py

Conversational CLI mood-to-music journey using the Nepali dataset.

Run with:  uv run python src/pipeline.py
       or: make run  (from project root)
"""

import sys
import joblib
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from recommendation.mood_parser import parse_mood
from recommendation.recommender import recommend

DATASET_PATH = Path("data/processed/nepali_dataset.csv")
MODELS_DIR = Path("models")

BANNER = """
╔══════════════════════════════════════════════════╗
║          🎵  Music Mood Therapy  🎵               ║
║   Tell me how you feel. I'll find your path.    ║
╚══════════════════════════════════════════════════╝
"""

SEP = "─" * 54
SONG_LABELS = ["MATCHING YOUR MOOD", "THE BRIDGE", "YOUR DESTINATION"]
POSITIVE_WORDS = {
    "good",
    "great",
    "better",
    "much better",
    "amazing",
    "awesome",
    "happy",
    "fantastic",
    "fine",
    "wonderful",
    "yes",
    "yeah",
    "yep",
    "lifted",
    "lighter",
    "nice",
    "improved",
    "refreshed",
}


def is_feeling_better(text: str) -> bool:
    return any(word in text.lower().strip() for word in POSITIVE_WORDS)


def print_songs(emotion: str, songs: list[dict]) -> None:
    print(f"\n  Detected emotion: {emotion}")
    print("\n  Here's your journey to a better place:\n")
    print(SEP)
    for song, label in zip(songs, SONG_LABELS):
        print(f"\n  {label}")
        print(f"  {song['track_name']}")
        print(f"  {song['artists']}")
        print(f"  Valence: {song['valence']}  |  Energy: {song['energy']}")
        print(f"  {song['spotify_url']}")
    print(f"\n{SEP}")


def run() -> None:
    print(BANNER)

    mlp_path = MODELS_DIR / "emotion_classifier_mlp_nepali.pkl"
    for path, label in [
        (DATASET_PATH, "Nepali dataset"),
        (mlp_path, "Nepali MLP model"),
    ]:
        if not path.exists():
            print(f"  Error: {label} not found.")
            print("  Run 'make train' first to set up the data pipeline.")
            sys.exit(1)

    print("  Loading song database...", end=" ", flush=True)
    df = pd.read_csv(DATASET_PATH)
    print(f"done  ({len(df):,} Nepali songs)")

    print("  Loading Nepali MLP neural network...", end=" ", flush=True)
    mlp = joblib.load(mlp_path)
    print("done\n")

    while True:
        print("  How are you feeling right now?")
        print("  (type 'quit' to exit)\n")
        user_input = input("  > ").strip()

        if user_input.lower() in {"quit", "exit", "q"}:
            print("\n  Take care. Come back whenever you need it. 🎵\n")
            break

        if not user_input:
            print("\n  Tell me a bit more about how you're feeling.\n")
            continue

        print("\n  Detecting your mood...", end=" ", flush=True)
        emotion = parse_mood(user_input)
        songs = recommend(emotion, df, mlp_model=mlp)
        print("done\n")

        print_songs(emotion, songs)
        print("\n  Listen through them in order.")
        print("  How are you feeling now?\n")
        follow_up = input("  > ").strip()

        if is_feeling_better(follow_up):
            print("\n  Glad to hear it. Music has a way of doing that. 🎵\n")
            continue

        print("\n  Let's try another set.\n")
        print("  Detecting your mood...", end=" ", flush=True)
        emotion2 = parse_mood(follow_up)
        songs2 = recommend(emotion2, df, mlp_model=mlp)
        print("done\n")
        print_songs(emotion2, songs2)
        print("\n  Take your time with these. 🎵\n")


if __name__ == "__main__":
    run()
