"""
api.py — FastAPI REST API server

Two features:
  POST /api/recommend  — mood text → 3 Nepali songs via ISO Principle
  POST /api/classify   — audio file upload → emotion probabilities from MLP

Run with:  cd backend && uv run uvicorn src.api:app --reload --port 8000
       or: make api (from project root)
"""

import logging
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from recommendation.mood_parser import parse_mood
from recommendation.recommender import recommend
from classification.classifier import classify_audio

# ── Loguru format ──────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<dim>{time:HH:mm:ss}</dim> "
        "<level>{level: <5}</level> "
        "<cyan>{name}</cyan> "
        "<dim>›</dim> "
        "{message}"
    ),
    level="INFO",
    colorize=True,
)


# ── Silence uvicorn's /health access log lines ────────────────────────────────
# Docker healthcheck fires every 10s — no value in logging it.
class _HealthFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "GET /health" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_HealthFilter())

app = FastAPI(title="Music Mood Therapy API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://localhost",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

DATASET_PATH = Path(__file__).parent.parent / "data/processed/nepali_dataset.csv"
MLP_PATH = Path(__file__).parent.parent / "models/emotion_classifier_mlp_nepali.pkl"

ALLOWED_AUDIO = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


@app.on_event("startup")
async def load_resources() -> None:
    for path in (DATASET_PATH, MLP_PATH):
        if not path.exists():
            logger.error(f"Required file not found: {path}")
            raise RuntimeError(
                f"Required file not found: {path}\n"
                "Run 'make train' from the project root first."
            )

    app.state.df = pd.read_csv(DATASET_PATH)
    app.state.mlp = joblib.load(MLP_PATH)

    logger.success(f"Dataset loaded    {len(app.state.df):,} songs")
    logger.success(f"MLP loaded        {MLP_PATH.name}")
    logger.success("API ready ─────── :8000")


# ── Models ────────────────────────────────────────────────────────────────────


class MoodRequest(BaseModel):
    text: str


class Song(BaseModel):
    track_name: str
    artists: str
    emotion: str
    valence: float
    energy: float
    track_id: str
    spotify_url: str


class RecommendResponse(BaseModel):
    emotion: str
    songs: list[Song]


class ClassifyResponse(BaseModel):
    predictions: dict[str, float]  # emotion → probability (all 6, sum to 1.0)
    top_emotion: str  # highest probability emotion


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/api/recommend", response_model=RecommendResponse)
def recommend_songs(req: MoodRequest, request: Request) -> RecommendResponse:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Mood text cannot be empty.")

    client = request.client.host if request.client else "unknown"
    logger.info(
        f"▶ recommend  '{req.text[:60]}{'…' if len(req.text) > 60 else ''}'  [{client}]"
    )

    t0 = time.perf_counter()

    emotion = parse_mood(req.text)
    songs = recommend(emotion, app.state.df, mlp_model=app.state.mlp)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.success(f"✓ recommend  {emotion} → {len(songs)} songs  ({elapsed_ms:.0f}ms)")

    return RecommendResponse(emotion=emotion, songs=songs)


@app.post("/api/classify", response_model=ClassifyResponse)
async def classify_song(file: UploadFile = File(...)) -> ClassifyResponse:
    suffix = Path(file.filename or "audio.mp3").suffix.lower()
    if suffix not in ALLOWED_AUDIO:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_AUDIO)}",
        )

    logger.info(f"▶ classify   '{file.filename}'")

    t0 = time.perf_counter()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        predictions = classify_audio(tmp_path, app.state.mlp)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if predictions is None:
        raise HTTPException(
            status_code=422,
            detail="Could not extract audio features. Is the file a valid audio file?",
        )

    top_emotion = max(predictions, key=lambda k: predictions[k])
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.success(
        f"✓ classify   {top_emotion} ({predictions[top_emotion]:.0%})  ({elapsed_ms:.0f}ms)"
    )

    return ClassifyResponse(predictions=predictions, top_emotion=top_emotion)


@app.get("/health")
async def health() -> dict:
    songs_loaded = len(app.state.df) if hasattr(app.state, "df") else 0
    model_status = "MLP loaded" if hasattr(app.state, "mlp") else "not loaded"
    return {"status": "ok", "songs_loaded": songs_loaded, "model": model_status}
