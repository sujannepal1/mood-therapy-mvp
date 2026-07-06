# Music Mood Therapy — Project Roadmap & System Design

> **Core concept:** User describes their mood → OpenAI classifies emotion → MLP neural network classifies each recommended song live → 3 songs guide them from that emotional state toward Happy via the ISO Principle from clinical music therapy.

---

## Status

- [x] Dataset downloaded — Kaggle "Spotify Tracks Dataset" (89,740 unique songs after dedup)
- [x] GitHub repo — `github.com/prasannakoirala58/mood-music-therapy`
- [x] Phase 0 — Scaffold, Docker, uv, folder structure
- [x] Phase 1 — Data labeling + RF + MLP training
- [x] Phase 2 — OpenAI mood parser + ISO recommender + live MLP inference
- [x] Phase 3 — Conversational CLI pipeline
- [x] Phase 4 — Librosa audio demo
- [x] Phase 5 — React + TypeScript frontend + FastAPI REST API

---

## 1. System Architecture

```
BROWSER / CLI
  User: "I feel completely empty and nothing excites me"
                        │
                        ▼
              ┌──────────────────┐
              │  OpenAI API      │  GPT-4o-mini, temperature=0
              │  mood_parser.py  │  Strict prompt → one of 6 labels
              └────────┬─────────┘
                       │  "Sad"
                       ▼
              ┌──────────────────┐
              │  ISO Engine      │  Maps "Sad" → (0.20, 0.20)
              │  recommender.py  │  Calculates 3 waypoints
              └────────┬─────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    (0.20, 0.20)  (0.50, 0.50)  (0.80, 0.80)
    match mood     bridge        destination
         │             │             │
         └─────────────┼─────────────┘
                       ▼
              ┌──────────────────┐
              │  89k Song CSV    │  Euclidean nearest-neighbour
              │  dataset_labeled │  Sample from top-K pool (variety)
              └────────┬─────────┘
                       │  3 candidate songs
                       ▼
              ┌──────────────────┐
              │  MLP Neural Net  │  Live inference on each song's
              │  (mlp.predict)   │  8 audio features → emotion label
              └────────┬─────────┘
                       ▼
         Song 1 — Matching your mood   (MLP emotion + Spotify link)
         Song 2 — The bridge           (MLP emotion + Spotify link)
         Song 3 — Your destination     (MLP emotion + Spotify link)
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
   FastAPI → React UI          CLI terminal
   http://localhost:5173        make run
```

---

## 2. ML Architecture

### Why two models?
We train both and compare them side-by-side. This is the academic rigour:
- RF is the gold-standard tabular baseline — hard to beat
- MLP is the neural network — shows you can build and explain one
- Either result has a story: "neural net won" OR "explain why RF beat it"

### Features (8 total)
```
valence        — musical positiveness (0.0–1.0)
energy         — intensity and activity (0.0–1.0)
tempo          — beats per minute
mode           — major (1) or minor (0) key
danceability   — how suitable for dancing (0.0–1.0)
loudness       — overall loudness in dB (−60–0)
acousticness   — confidence it's acoustic (0.0–1.0)
genre_encoded  — 125 genres as integers (via LabelEncoder)
```

### Model 1 — Random Forest (baseline)
```
100 decision trees, majority vote
class_weight='balanced'  — handles class imbalance
Scale-invariant (no preprocessing needed)
Gives feature importance chart for the presentation slide
Actual accuracy: 100% (label leakage — learns back its own rules)
Defense talking point: "circularity is a known limitation of rule-based labeling"
```

### Model 2 — MLP Neural Network
```
Input:   8 features
Hidden:  128 neurons → 64 neurons (ReLU)
Output:  6 neurons (softmax — one per emotion)
Pipeline: StandardScaler → MLPClassifier
early_stopping=True, validation_fraction=0.1
Actual accuracy: 99.26% (stopped at iteration 39)
USED LIVE: runs inference on every recommended song during recommendations
```

### Overfitting — handled
```
Class imbalance   →  class_weight='balanced'
MLP overfitting   →  early_stopping + validation_fraction=0.1
Feature scale     →  StandardScaler in MLP Pipeline
Evaluation        →  80/20 stratified train/test split
Label noise       →  Rule-based labels acknowledged as limitation in defense
```

---

## 3. Emotion Mapping — Russell's Circumplex Model

```
                   HIGH ENERGY (1.0)
                        │
        Angry           │    Surprise
       (0.20, 0.80)     │   (0.60, 0.85)
                        │
NEGATIVE ───────────────┼─────────────────── POSITIVE
(0.0)                   │                   (1.0)
                   Fear │         Happy
               (0.20,0.60)   (0.80, 0.80) ← TARGET
                        │
        Disgust         │
       (0.15, 0.50)     │
                        │
        Sad             │
       (0.20, 0.20)     │
                   LOW ENERGY (0.0)
```

### Emotion labeling rules (`label_emotions.py`)
```python
Happy:    v > 0.6 AND e > 0.6
Sad:      v < 0.4 AND e < 0.4
Angry:    v < 0.4 AND e > 0.65
Fear:     v < 0.5 AND 0.35 <= e <= 0.7 AND mode == 0  (minor key)
Disgust:  v < 0.35 AND e < 0.5
Surprise: v >= 0.5 AND e >= 0.65          ← real zone, not catch-all
Mid-zone: assigned to nearest centroid     ← no song is discarded
```

### ISO waypoints — how the 3-song path is calculated
```python
waypoints = [
    (current_v, current_e),                               # match
    ((current_v + 0.80) / 2, (current_e + 0.80) / 2),   # bridge
    (0.80, 0.80),                                         # destination
]
# Each waypoint → nearest-neighbour search → MLP classifies the result song
```

---

## 4. Spotify Integration — Zero API Calls

The Kaggle dataset includes `track_id` — the Spotify track identifier.

```python
spotify_url = f"https://open.spotify.com/track/{row['track_id']}"
```

No API key. No authentication. Works instantly. Spotify Recommendations API was
deprecated for apps created after late 2024 — this approach has no dependency on it.

---

## 5. Variety — Why Songs Differ Each Session

Instead of always picking the single nearest song to each waypoint, the recommender
samples randomly from the top-K closest songs:

```
Waypoint 1 (match):       top 30 → random pick
Waypoint 2 (bridge):      top 20 → random pick
Waypoint 3 (destination): top 10 → random pick
```

This means every session produces a different journey while still staying close to
the target emotional coordinate. Academic framing: "stochastic nearest-neighbour
selection balances precision and exploration."

---

## 6. Tech Stack

| Tool | Version | Why |
|------|---------|-----|
| Python | 3.11 | Stable, excellent ML support |
| uv | latest | 10–100× faster than pip, deterministic lockfile |
| Docker + compose | latest | Reproducible, one-command startup |
| FastAPI | ^0.111 | Modern async REST, auto OpenAPI docs |
| pandas | ^2.2 | CSV handling, vectorised nearest-neighbour |
| scikit-learn | ^1.4 | RF + MLP + StandardScaler + LabelEncoder |
| numpy | ^1.26 | Euclidean distance calculations |
| openai | ^1.30 | GPT-4o-mini for mood text classification |
| librosa | ^0.10 | Audio feature extraction (demo only) |
| joblib | ^1.4 | Save/load .pkl model files |
| python-dotenv | ^1.0 | Load .env at runtime |
| React | 18.3 | Component-based frontend |
| TypeScript | ^5 | Strict typing throughout frontend |
| Tailwind CSS | ^3.4 | Utility-first styling |
| Vite | ^5.3 | Fast dev server + production build |
| framer-motion | ^11 | Song card entrance animations |
| lucide-react | ^0.400 | Icon set (Spotify, arrows, etc.) |

---

## 7. API Contract

### `POST /api/recommend`
```json
Request:
{ "text": "I feel completely empty and hopeless" }

Response:
{
  "emotion": "Sad",
  "songs": [
    {
      "track_name": "string",
      "artists": "string",
      "emotion": "Sad",        ← MLP prediction (live inference)
      "valence": 0.199,
      "energy": 0.200,
      "track_id": "string",
      "spotify_url": "https://open.spotify.com/track/..."
    },
    { ... },   ← bridge song
    { ... }    ← destination song
  ]
}
```

### `GET /health`
```json
{ "status": "ok", "songs_loaded": 89740, "model": "MLP loaded" }
```

---

## 8. Defense Talking Points

**"What neural network did you build?"**
An MLP (Multi-Layer Perceptron) with two hidden layers — 128 neurons then 64 neurons, ReLU activations, trained with backpropagation on 8 Spotify audio features from 89,740 songs. The model runs live during every recommendation: when a song is selected, its features are fed into the MLP and it predicts the emotion in real time.

**"What is the ISO Principle?"**
A technique from clinical music therapy: match the listener's current emotional state first, then gradually guide them toward the target emotion. You cannot jump straight from sad to happy — the emotional dissonance makes it worse. Three songs, three waypoints.

**"What is Russell's Circumplex Model?"**
A 2D psychological model plotting emotions on two axes: valence (positive vs negative) and arousal (high vs low energy). Published 1980, widely used in affective computing. It's how we convert emotion labels into numerical coordinates.

**"Why 100% accuracy on the Random Forest?"**
The emotion labels were created from the same audio features the RF trains on. So the model is learning back its own rules — that's label circularity. It's a known limitation of rule-based labeling. Human-annotated ground truth would give realistic accuracy. We acknowledge this and it's why we focus on the MLP's live inference during the demo.

**"How does the mood detection work?"**
User text → GPT-4o-mini with a strict classification prompt → exactly one of 6 emotion labels, returned in under 200ms. `temperature=0` ensures the same input always gives the same output.

**"How do you prevent overfitting in the MLP?"**
Three mechanisms: `StandardScaler` normalises inputs before training, `early_stopping=True` halts when validation accuracy stops improving, and `validation_fraction=0.1` holds back 10% of training data as an internal check.

**"What are the limitations?"**
Emotion labeling is rule-based (not human-annotated). The Surprise and Disgust categories overlap in acoustic space. Emotion is subjective — the same song hits differently for different people. The recommendation catalog is fixed at 89,740 songs from one Kaggle dataset.

---

## 9. Key Decisions Log

| Decision | Choice | Reason |
|----------|--------|--------|
| Package manager | `uv` | Fast, deterministic, modern |
| LLM | GPT-4o-mini | Cheap, fast, excellent at classification |
| Song discovery | 114k Kaggle catalog + Spotify URLs | Recommendations API deprecated |
| Frontend | React + TypeScript | Proper engineering stack, not Streamlit |
| REST API | FastAPI | Async, automatic docs, Pydantic validation |
| Neural network | sklearn MLPClassifier | Real network you own and can explain |
| Baseline | Random Forest | Industry-standard tabular baseline |
| Live inference | MLP predicts song emotion at request time | Model does real work in production |
| Variety | Sample top-K nearest (not always top-1) | Different songs each session |
| Target emotion | Happy (0.80, 0.80) | Simplest, most defensible, clinically grounded |
| Surprise label | Real zone (v≥0.5 AND e≥0.65), not catch-all | Better label accuracy in bridge zone |
