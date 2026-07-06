# CLAUDE.md — Music Mood Therapy Project

This file is the source of truth for how Claude Code should behave in this project.
**Read this before touching any file. Ask before you build anything not explicitly described here.**

---

## What This Project Is

A Music Mood Therapy recommendation engine built as a Data Analytics academic project.

**Core concept:** User describes their mood → OpenAI classifies their emotion → MLP neural network classifies each recommended song live → 3 songs guide them from that emotional state toward Happy via the **ISO Principle** from clinical music therapy (match → bridge → target).

**Academic backbone:** Russell's Circumplex Model (1980), Ekman's 6 basic emotions, ISO Principle from clinical music therapy.

---

## What The App Has

| Feature | Status | Notes |
|---------|--------|-------|
| Mood → 3 song recommendation | Live | ISO Principle, GPT-4o-mini, Nepali dataset |
| Audio file classify | Live | Upload MP3/WAV → MLP → emotion probability bars |
| Two-tab React frontend | Live | "Find My Path" + "Classify a Song" |
| MLP quality gate | Live | Up to 3 retries if MLP prediction mismatches expected emotion |
| Live MLP inference | Live | Every recommended song classified at request time |
| Loguru structured logging | Live | Compact colored format, health check silent |
| Docker full-stack | Live | `docker compose up --build` |
| CLI pipeline | Live | `make run` |
| Librosa audio demo | Live | `make demo FILE=path/to/song.mp3` |

## What The App Does NOT Have

- Spotify API calls (their Recommendations API was deprecated for apps created after late 2024)
- Genre feature in the Nepali model (only 7 features — no genre labels in Nepali data)
- Audio preview / playback in the browser
- User accounts, history, or any persistence
- The Kaggle 89k-song dataset in production (those models exist as `*_rf.pkl` / `*_mlp.pkl` but the API uses `*_nepali.pkl` only)
- Streamlit (frontend is React + TypeScript — never suggest Streamlit)

---

## Tech Stack — Non-Negotiable

| Layer | Tool | Notes |
|-------|------|-------|
| Language | Python 3.11 | Pinned in `backend/.python-version` |
| Package manager | `uv` | NOT pip, NOT poetry, NOT conda — ever |
| Containers | Docker + docker-compose | Full stack via `docker compose up --build` |
| LLM | OpenAI GPT-4o-mini | `OPENAI_API_KEY` in `.env` |
| ML | scikit-learn RF + MLPClassifier | Both trained; MLP runs live in recommendations |
| REST API | FastAPI + uvicorn | `backend/src/api.py`, port 8000 |
| Logging | Loguru | Compact colored format, configured in `api.py` startup |
| Frontend | React 18 + TypeScript + Tailwind CSS + Vite | NOT Streamlit, NOT plain JSX |
| Frontend runtime | Node 20 | Dockerfile in `frontend/` |
| Audio collection | yt-dlp + librosa | Downloads full audio from YouTube, extracts 7 features |
| Entry point (CLI) | `make run` | Trains if needed, then runs CLI |
| Entry point (web) | `make api` + `make frontend` | Two terminals, or `docker compose up` |

**Never suggest `pip install`.** Always `uv add <package>` from `backend/`.
**Never suggest Streamlit.** Frontend is React + TypeScript in `frontend/src/`.
**Frontend source files use `.tsx` and `.ts` — never `.jsx` or `.js`.**

---

## Dataset — Nepali Songs (Production)

The API and CLI run on a **Nepali song dataset**, NOT the 89k Kaggle dataset.

| File | Songs | Source | Labels |
|------|-------|--------|--------|
| `nepali_dataset.csv` | 70 | Collected via yt-dlp from 6 Spotify emotion playlists | Human-labeled by playlist category |
| `nepali_dataset_500.csv` | 303 | Teammate's `features.csv` (597 songs locally recorded) | Rule-labeled using same thresholds as `label_emotions.py` |
| **Combined training set** | **373** | Both files merged, deduped by track_id | — |
| Train / test split | 298 / 75 | 80/20, stratified by emotion | — |

**The Kaggle dataset** (`dataset.csv`, 89k songs) and its models (`emotion_classifier_rf.pkl`, `emotion_classifier_mlp.pkl`, `genre_label_encoder.pkl`) still exist in models/ but are NOT used by the API or CLI.

---

## ML Architecture

### Features — 7 total (no genre)
```
valence        — computed: 0.40 * mode + 0.35 * tempo_norm + 0.25 * centroid_norm
energy         — computed from RMS energy via librosa
tempo          — BPM from librosa beat tracking
mode           — major (1) / minor (0) via Krumhansl-Schmuckler key profiles
danceability   — computed from tempo regularity
loudness       — RMS-based loudness in dB
acousticness   — harmonic-to-percussive energy ratio
```

### Model performance on 75 held-out test songs
```
Random Forest  — 74.67%  (100 trees, class_weight='balanced')
MLP Neural Net — 66.67%  (32 → 16 ReLU, StandardScaler, early_stopping)
```
Architecture is deliberately sized to the dataset: 32→16 for 373 songs vs 128→64 (used for the 89k Kaggle set).
These are realistic accuracies — NOT the 99%+ you get with Kaggle label circularity.
Sad (11 songs) and Disgust (8 songs) classify at 0% — a data scarcity problem, not a model problem.

### Saved model files
```
models/emotion_classifier_rf_nepali.pkl   ← Nepali RF (reference / comparison only)
models/emotion_classifier_mlp_nepali.pkl  ← Nepali MLP (used live by API + CLI)
models/emotion_classifier_rf.pkl          ← Kaggle RF (not used in production)
models/emotion_classifier_mlp.pkl         ← Kaggle MLP (not used in production)
models/genre_label_encoder.pkl            ← Kaggle encoder (not used in production)
```

---

## Project Structure — Current State

```
music-therapy/
│
├── backend/
│   ├── src/
│   │   ├── api.py                        ← FastAPI REST server (POST /api/recommend, POST /api/classify)
│   │   ├── pipeline.py                   ← Conversational CLI loop
│   │   ├── train_classifier.py           ← Trains RF + MLP on Nepali data → saves *_nepali.pkl
│   │   ├── label_emotions.py             ← Rule-based labeling for Kaggle dataset (not for Nepali)
│   │   ├── collect_spotify_data.py       ← yt-dlp + librosa: downloads Nepali songs → nepali_dataset.csv
│   │   ├── process_features_csv.py       ← Converts features.csv → nepali_dataset_500.csv (303 songs)
│   │   ├── common/
│   │   │   └── features.py              ← Shared librosa extraction (7 features)
│   │   ├── recommendation/
│   │   │   ├── mood_parser.py           ← OpenAI: text → one of 6 emotion labels
│   │   │   └── recommender.py           ← ISO engine: waypoints, nearest-neighbour, MLP quality gate
│   │   └── classification/
│   │       └── classifier.py            ← Audio path + MLP → emotion probability dict
│   │
│   ├── data/
│   │   ├── raw/
│   │   │   ├── dataset.csv              ← Kaggle 89k songs (git-ignored, not used in production)
│   │   │   └── features.csv            ← Teammate's 597-song feature extraction (git-ignored)
│   │   └── processed/
│   │       ├── dataset_labeled.csv      ← Labeled Kaggle data (git-ignored, generated)
│   │       ├── nepali_dataset.csv       ← 70 Nepali songs, human-labeled (git-ignored, production)
│   │       └── nepali_dataset_500.csv   ← 303 Nepali songs, rule-labeled (git-ignored, generated)
│   │
│   ├── models/                          ← .pkl files (git-ignored, generated)
│   ├── demo/librosa_demo.py             ← Raw audio → features → recommend (standalone demo)
│   ├── pyproject.toml
│   ├── uv.lock
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx                      ← Two-tab layout with framer-motion tab pill animation
│   │   ├── main.tsx
│   │   ├── types.ts                     ← Song, RecommendResult, ClassifyResult interfaces
│   │   ├── components/
│   │   │   ├── MoodInput.tsx            ← Text input with submit
│   │   │   ├── ResultsPanel.tsx         ← 3-song ISO journey display
│   │   │   ├── SongCard.tsx             ← Song card + Spotify button
│   │   │   ├── ResonanceMap.tsx         ← SVG: dot position on Russell's Circumplex
│   │   │   └── ClassifyTab.tsx          ← Drag-and-drop upload + animated probability bars
│   │   ├── hooks/
│   │   │   ├── useRecommend.ts          ← POST /api/recommend
│   │   │   └── useClassify.ts           ← POST /api/classify (validates file, posts FormData)
│   │   └── lib/emotions.ts             ← Emotion colors, metadata, DEFAULT_ACCENT
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── .env.local                       ← VITE_API_URL (git-ignored)
│   └── Dockerfile
│
├── Makefile
├── docker-compose.yml
├── .env                                 ← NEVER committed
├── .env.example
├── .gitignore
├── README.md
├── ROADMAP.md
└── CLAUDE.md                            ← This file
```

---

## Execution Flow

```
SETUP (run once per machine):
  make train
    └─ train_classifier.py reads nepali_dataset.csv + nepali_dataset_500.csv
    └─ trains RF + MLP → emotion_classifier_rf_nepali.pkl + emotion_classifier_mlp_nepali.pkl

  (Optional — only needed if re-collecting Nepali songs from scratch)
  make collect   →  yt-dlp download + librosa extraction → nepali_dataset.csv
  make process   →  features.csv → nepali_dataset_500.csv

WEB APP (two terminals):
  Terminal 1:  make api          →  FastAPI on http://localhost:8000
  Terminal 2:  make frontend     →  React   on http://localhost:5173

  POST /api/recommend flow:
    1. text → OpenAI GPT-4o-mini → emotion label
    2. emotion → ISO 3 waypoints on Russell's Circumplex
    3. each waypoint → nearest-neighbour in nepali_dataset.csv (sampled from top-K)
    4. song features → MLP.predict() → live emotion label
    5. mismatch → retry up to 3 times (quality gate)
    6. return { emotion, songs[3] }

  POST /api/classify flow:
    1. audio file → tempfile
    2. librosa → 7 features
    3. MLP.predict_proba() → 6 emotion probabilities
    4. return { predictions, top_emotion }

CLI APP:
  make run   →  pipeline.py  (Nepali dataset + Nepali MLP)

FULL STACK VIA DOCKER:
  docker compose up --build
```

---

## Build Phases — Completed

- [x] Phase 0 — Scaffold, Docker, uv, folder structure, GitHub
- [x] Phase 1 — Kaggle data labeling (`label_emotions.py`) + RF + MLP training on 89k songs
- [x] Phase 2 — Mood parser + ISO recommender with live MLP inference
- [x] Phase 3 — Conversational CLI pipeline
- [x] Phase 4 — Librosa audio demo
- [x] Phase 5 — React + TypeScript frontend + FastAPI REST API
- [x] Phase 6 — Loguru structured logging (compact colored, health silent)
- [x] Phase 7 — Nepali dataset collection (yt-dlp + librosa, 70 songs from 6 playlists)
- [x] Phase 8 — Teammate's features.csv → nepali_dataset_500.csv (303 rule-labeled songs)
- [x] Phase 9 — Backend restructure: common/ + recommendation/ + classification/ subpackages
- [x] Phase 10 — Nepali MLP + RF trained on combined 373 songs (7 features, no genre)
- [x] Phase 11 — "Classify a Song" tab (drag-and-drop upload → animated probability bars)

---

## Security Practices

- All API keys live in `.env` only — never hardcoded, never printed, never logged.
- `.env` is in `.gitignore` and must never be committed.
- CORS locked to `localhost:5173` and `localhost:3000` — no wildcard `*`.
- All request bodies validated by Pydantic.
- The API is stateless — no user data stored anywhere.
- Temp files from `/api/classify` are deleted immediately after classification.
- OpenAI call uses `temperature=0` — deterministic.
- Model `.pkl` files loaded with `joblib` — never unpickle untrusted sources.

---

## Rules for Claude Code

### 1. Ask before building. Never assume.
If a decision is not documented here, **stop and ask Prasanna before implementing**.
He must be able to understand and defend every decision at assessment. No surprises.
This includes: new files, new endpoints, new features, new dependencies, threshold changes, hyperparameter changes, restructuring.

### 2. Work like a senior engineer, not a code generator.
Explain WHY before writing code. Flag tradeoffs. Simple and readable beats clever.
Don't add code for hypothetical future scenarios.

### 3. No silent decisions.
If you change anything previously decided (a threshold, a feature, a file path), say so explicitly and explain why.

### 4. Keep implementation lean.
No comments explaining WHAT — good names do that. No unused imports. No dead code.

### 5. Data and model files are git-ignored.
Never suggest committing `dataset.csv`, `features.csv`, any `*_labeled.csv`, `nepali_dataset*.csv`, or any `.pkl` file.

### 6. uv is the package manager — without exception.
Add a dependency: `cd backend && uv add <package>`. Never `pip install`.

### 7. Frontend is React + TypeScript. Full stop.
All frontend files are `.tsx` / `.ts`. Never suggest Streamlit or a different framework.

### 8. Six emotions only. Target is always Happy.
`Happy, Sad, Angry, Fear, Disgust, Surprise` — never add or remove labels.
ISO journey always ends at `(valence=0.80, energy=0.80)`.

### 9. Commits are Prasanna's — no co-author line.
When asked to commit and push: do it, but never add a `Co-Authored-By` line.

### 10. Do not hallucinate Nepali song or artist names.
You don't know Nepali music well enough. When test songs are needed, tell the user to use songs they already know and have on their machine.

---

## Key Context for Every Response

- **Audience:** University assessor. Code must be clean, explainable, and defensible.
- **Student:** Data Analytics student — explain decisions in plain language.
- **Production dataset:** 70 Nepali songs (`nepali_dataset.csv`) + 303 rule-labeled supplementary.
- **Spotify links are free:** `https://open.spotify.com/track/{track_id}` — no API calls needed.
- **MLP is live:** Neural network runs inference on every recommended song at request time.
- **Variety by design:** Sampled from top-K nearest (MATCH top_k=20, BRIDGE top_k=15, DESTINATION top_k=10).
- **Realistic accuracy:** 74.67% RF / 69.33% MLP on Nepali test songs — not inflated Kaggle numbers.
