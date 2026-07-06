#!/bin/sh
set -e

DATASET="/app/data/processed/nepali_dataset.csv"
MODEL="/app/models/emotion_classifier_mlp_nepali.pkl"

if [ ! -f "$DATASET" ]; then
    echo ""
    echo "  ERROR: Dataset not found."
    echo "  Run 'make collect' first to pull songs from Spotify,"
    echo "  then re-run 'docker compose up --build'."
    echo ""
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo ""
    echo "  Models not found — training now (~1 minute)..."
    echo ""
    cd /app && uv run python src/train_classifier.py
    echo ""
fi

exec "$@"
