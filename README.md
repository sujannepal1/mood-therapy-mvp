# Music Mood Therapy

A mood-to-music recommendation engine grounded in clinical music therapy and machine learning — built as a Data Analytics academic project.

> Tell me how you feel. I'll find three Nepali songs that move you from where you are to somewhere better.

---

## How It Works

The system applies the **ISO Principle** from clinical music therapy: you cannot jump from sad to happy — the emotional gap creates dissonance. Instead, music therapy starts by *matching* the listener's current state, then *bridges*, then arrives at the *destination*.

```
You type:  "I feel completely empty and hopeless"

Step 1 — OpenAI GPT-4o-mini classifies your mood
         → "Sad"

Step 2 — ISO engine maps "Sad" to coordinates on Russell's Circumplex Model
         → (valence=0.20, energy=0.20)
         → Calculates 3 waypoints toward Happy (0.80, 0.80)

Step 3 — Nearest-neighbour search across 70 Nepali songs
         → Samples from top-K closest songs (different result each session)

Step 4 — MLP neural network classifies each song live
         → Predicts emotion from 7 audio features extracted with librosa

Step 5 — You get three songs + Spotify links:
         [1] matching your mood    — meets you where you are
         [2] the bridge            — starts shifting the energy
         [3] your destination      — where you're headed
```

---

## Two Features

### Tab 1 — Find My Path
Type how you're feeling. Get three Nepali songs that guide you toward a happier state using the ISO Principle.

### Tab 2 — Classify a Song
Drag and drop any MP3/WAV/FLAC file. The MLP neural network analyses it with librosa and shows the full emotion breakdown — how much Happy, Sad, Angry, Fear, Disgust, and Surprise is in that song.

---

## Academic Foundation

| Concept | Source |
|---------|--------|
| ISO Principle | Clinical music therapy — match emotional state before guiding |
| Russell's Circumplex Model | Russell (1980) — 2D emotion space: valence × arousal |
| Ekman's 6 basic emotions | Happy, Sad, Angry, Fear, Disgust, Surprise |
| Audio features | 7 features computed via librosa + Krumhansl-Schmuckler key profiles |

---

## ML Architecture

Two classifiers trained on **373 Nepali songs** (70 hand-collected from Spotify playlists via yt-dlp, 303 from teammate's local recordings). Both answer: **"Given these 7 audio features, what emotion is this song?"**

```
Features (7 total — no genre):
  valence        computed: 0.40×mode + 0.35×tempo_norm + 0.25×centroid_norm
  energy         RMS energy via librosa
  tempo          BPM via librosa beat tracking
  mode           major(1) / minor(0) via Krumhansl-Schmuckler key profiles
  danceability   tempo regularity
  loudness       RMS-based dB
  acousticness   harmonic-to-percussive energy ratio

Random Forest (baseline)
  100 decision trees · class_weight='balanced'
  Accuracy: 74.67% on 75 held-out Nepali test songs

MLP Neural Network  ← used live in recommendations and classify
  Input:  7 features → StandardScaler
  Hidden: 32 → 16 neurons (ReLU)
  Output: 6 emotions (softmax)
  early_stopping + validation_fraction=0.20
  Accuracy: 66.67% on 75 held-out Nepali test songs
```

These are **realistic** accuracies on unseen Nepali songs — not the 99%+ you get when training on rule-labeled Kaggle data where the features and labels are circular.

### MLP Quality Gate
During recommendation, if the MLP's live prediction for a song doesn't match the ISO waypoint's expected emotion, the recommender retries with the next nearest song — up to 3 attempts. This keeps the therapeutic journey coherent.

---

## Setup

### Prerequisites
- Python 3.11+ and [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Node 20+
- Docker Desktop (for containerised run)
- OpenAI API key

### 1. Clone and configure
```bash
git clone git@github.com:prasannakoirala58/mood-music-therapy.git
cd music-therapy
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### 2. Install backend dependencies
```bash
cd backend && uv sync
```

### 3. Train the models (one-time, ~1 minute)
```bash
make train
# Trains RF + MLP on the Nepali dataset
# Saves emotion_classifier_rf_nepali.pkl and emotion_classifier_mlp_nepali.pkl
```

> **Note:** The Nepali datasets (`nepali_dataset.csv`, `nepali_dataset_500.csv`) must already exist in `backend/data/processed/`.
> If starting from scratch, run `make collect` and `make process` first (see below).

### 4. Run

**Full stack via Docker (recommended):**
```bash
docker compose up --build
# Backend: http://localhost:8000
# Frontend: http://localhost:5173
```

**Web app (two terminals):**
```bash
make api        # Terminal 1 — FastAPI backend on http://localhost:8000
make frontend   # Terminal 2 — React frontend on http://localhost:5173
```

**CLI:**
```bash
make run
```

---

## Project Structure

```
music-therapy/
├── backend/
│   ├── src/
│   │   ├── api.py                   ← FastAPI REST server (/api/recommend, /api/classify)
│   │   ├── pipeline.py              ← CLI conversation loop
│   │   ├── train_classifier.py      ← Trains RF + MLP on Nepali data
│   │   ├── label_emotions.py        ← Rule-based labeling for Kaggle data (one-time setup)
│   │   ├── collect_spotify_data.py  ← yt-dlp + librosa: builds nepali_dataset.csv
│   │   ├── process_features_csv.py  ← features.csv → nepali_dataset_500.csv
│   │   ├── common/
│   │   │   └── features.py         ← Shared librosa feature extraction (7 features)
│   │   ├── recommendation/
│   │   │   ├── mood_parser.py      ← OpenAI: text → emotion label
│   │   │   └── recommender.py      ← ISO engine + MLP quality gate
│   │   └── classification/
│   │       └── classifier.py       ← Audio file → MLP → emotion probabilities
│   ├── data/
│   │   ├── raw/                    ← dataset.csv + features.csv (git-ignored)
│   │   └── processed/              ← nepali_dataset.csv + nepali_dataset_500.csv (git-ignored)
│   ├── models/                     ← .pkl files (git-ignored)
│   └── demo/librosa_demo.py        ← Standalone audio demo
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx                 ← Two-tab layout (framer-motion tab animation)
│   │   ├── types.ts                ← TypeScript interfaces
│   │   ├── components/
│   │   │   ├── MoodInput.tsx       ← Mood text input
│   │   │   ├── ResultsPanel.tsx    ← 3-song ISO journey
│   │   │   ├── SongCard.tsx        ← Song card + Spotify link
│   │   │   ├── ResonanceMap.tsx    ← SVG dot on Russell's Circumplex
│   │   │   └── ClassifyTab.tsx     ← Drag-and-drop upload + probability bars
│   │   ├── hooks/
│   │   │   ├── useRecommend.ts     ← POST /api/recommend
│   │   │   └── useClassify.ts      ← POST /api/classify
│   │   └── lib/emotions.ts        ← Emotion colors + metadata
│   └── ...
│
├── Makefile
├── docker-compose.yml
└── ROADMAP.md                      ← System design + defense talking points
```

---

## Makefile Commands

```bash
# Setup
make install          # Install backend Python deps via uv

# Data pipeline (Nepali — primary)
make collect          # Download Nepali songs from YouTube via yt-dlp + extract features
make process          # Convert teammate's features.csv → nepali_dataset_500.csv
make train            # Train RF + MLP on combined Nepali datasets → *_nepali.pkl

# Data pipeline (Kaggle — original, for comparison models)
make label            # Label 89k Kaggle songs with emotions → dataset_labeled.csv
make train-kaggle     # Train RF + MLP on labeled Kaggle data → emotion_classifier_*.pkl

# Run the app
make run              # CLI conversation app (Nepali models must exist)
make start            # Smart start: trains first if models missing, then runs CLI
make api              # FastAPI backend on port 8000 (hot-reload)
make frontend         # React + Vite dev server on port 5173
make typecheck        # TypeScript strict type-check

# Docker
make up               # docker compose up --build (full stack)
make down             # docker compose down

# Demo + maintenance
make demo FILE=...    # Librosa audio demo: make demo FILE=song.mp3
make clean            # Remove generated data + model files
make help             # Show all commands
```

---

## API

### `POST /api/recommend`
```json
Request:  { "text": "I feel completely empty" }
Response: {
  "emotion": "Sad",
  "songs": [
    { "track_name": "...", "artists": "...", "emotion": "Sad",
      "valence": 0.20, "energy": 0.20,
      "track_id": "...", "spotify_url": "https://open.spotify.com/track/..." },
    { ... },
    { ... }
  ]
}
```

### `POST /api/classify`
```json
Request:  multipart/form-data — field "file" (MP3/WAV/FLAC/M4A/OGG, max 30MB)
Response: {
  "predictions": {
    "Happy": 0.12, "Sad": 0.08, "Angry": 0.05,
    "Fear": 0.07, "Disgust": 0.06, "Surprise": 0.62
  },
  "top_emotion": "Surprise"
}
```

---

## Built With

**Backend:** Python 3.11 · uv · FastAPI · scikit-learn · OpenAI · librosa · yt-dlp · pandas · numpy · joblib · loguru · python-multipart

**Frontend:** React 18 · TypeScript · Tailwind CSS · Vite · framer-motion · lucide-react

**Infrastructure:** Docker · docker-compose

**Dataset:** 70 Nepali songs collected from Spotify emotion playlists via yt-dlp + librosa, supplemented by 303 songs from a teammate's local audio collection.

**Original reference dataset:** [Kaggle — Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (89k tracks) — used to build the Kaggle baseline models only.
