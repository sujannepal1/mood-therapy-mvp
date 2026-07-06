"""
train_classifier.py

Trains two classifiers on the combined Nepali dataset and saves both to models/:
  - Random Forest      → emotion_classifier_rf_nepali.pkl
  - MLP Neural Network → emotion_classifier_mlp_nepali.pkl

No genre feature — the Nepali dataset doesn't have genre labels.
Features: valence, energy, tempo, mode, danceability, loudness, acousticness (7 total)

Data sources (combined):
  - data/processed/nepali_dataset.csv      (human-labeled via Spotify playlists)
  - data/processed/nepali_dataset_500.csv  (rule-labeled from teammate's features.csv)
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

NEPALI_PATH     = Path("data/processed/nepali_dataset.csv")
NEPALI_500_PATH = Path("data/processed/nepali_dataset_500.csv")
MODELS_DIR      = Path("models")

FEATURES = [
    "valence", "energy", "tempo", "mode",
    "danceability", "loudness", "acousticness",
]
TARGET = "emotion"


def load_data() -> pd.DataFrame:
    frames = []

    if NEPALI_PATH.exists():
        df1 = pd.read_csv(NEPALI_PATH)
        print(f"  nepali_dataset.csv       {len(df1):>4} songs  (human-labeled)")
        frames.append(df1)
    else:
        print(f"  WARNING: {NEPALI_PATH} not found — run collect_spotify_data.py first")

    if NEPALI_500_PATH.exists():
        df2 = pd.read_csv(NEPALI_500_PATH)
        print(f"  nepali_dataset_500.csv   {len(df2):>4} songs  (rule-labeled)")
        frames.append(df2)
    else:
        print(f"  WARNING: {NEPALI_500_PATH} not found — run process_features_csv.py first")

    if not frames:
        raise FileNotFoundError("No Nepali dataset files found. Run the data pipeline first.")

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="track_id")
    df = df.dropna(subset=FEATURES + [TARGET])
    return df


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def train_mlp(X_train, y_train) -> Pipeline:
    # 32→16 chosen deliberately for ~300 training samples.
    # 128→64 (used for the 89k Kaggle set) has ~9,700 parameters for 298 samples —
    # overfitting risk. 32→16 gives ~900 parameters, a healthier ratio.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.20,
            random_state=42,
            verbose=False,
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def print_feature_importances(rf: RandomForestClassifier) -> None:
    ranked = sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: x[1], reverse=True)
    print("\n  Feature importances (Random Forest):")
    for name, score in ranked:
        bar = "█" * int(score * 50)
        print(f"    {name:<20} {score:.4f}  {bar}")


def run() -> None:
    print("Loading Nepali datasets...")
    df = load_data()
    print(f"\n  Combined: {len(df):,} songs")
    print("\n  Emotion distribution:")
    for emotion, count in df[TARGET].value_counts().items():
        bar = "█" * int(count / 2)
        print(f"    {emotion:<10} {count:>4}  {bar}")

    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    print("Training Random Forest (100 trees)...")
    rf      = train_random_forest(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc   = (rf_preds == y_test).mean()
    print(f"  Accuracy: {rf_acc * 100:.2f}%")

    print("\nTraining Nepali MLP (32 → 16 → 6)...")
    mlp      = train_mlp(X_train, y_train)
    mlp_preds = mlp.predict(X_test)
    mlp_acc   = (mlp_preds == y_test).mean()
    n_iter    = mlp.named_steps["mlp"].n_iter_
    print(f"  Accuracy: {mlp_acc * 100:.2f}%  (stopped at iteration {n_iter})")

    print("\n" + "─" * 56)
    print(f"  RF  accuracy: {rf_acc * 100:.2f}%   MLP accuracy: {mlp_acc * 100:.2f}%")
    print(f"  Winner: {'Random Forest' if rf_acc >= mlp_acc else 'MLP Neural Network'}")
    print("─" * 56)

    print("\nRANDOM FOREST — per-class breakdown")
    print(classification_report(y_test, rf_preds, zero_division=0))
    print("MLP NEURAL NETWORK — per-class breakdown")
    print(classification_report(y_test, mlp_preds, zero_division=0))

    print_feature_importances(rf)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    rf_path  = MODELS_DIR / "emotion_classifier_rf_nepali.pkl"
    mlp_path = MODELS_DIR / "emotion_classifier_mlp_nepali.pkl"

    joblib.dump(rf,  rf_path)
    joblib.dump(mlp, mlp_path)

    print(f"\n  Saved → {rf_path}")
    print(f"  Saved → {mlp_path}")
    print("\nNepali model training complete. Run 'make api' to start the server.")


if __name__ == "__main__":
    run()
