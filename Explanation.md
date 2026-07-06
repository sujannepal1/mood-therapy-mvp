# Music Mood Therapy — How Everything Works

A plain-language guide to the entire project: what every file does, why it exists, and how the pieces connect.

---

## The Big Picture First

When you type "I've been feeling really low and nothing excites me", here is the exact journey your words take:

```
Your text
   ↓
OpenAI GPT-4o-mini reads it and returns one word: "Sad"
   ↓
"Sad" maps to coordinates on a 2D chart: valence=0.20, energy=0.20
   ↓
The system calculates 3 waypoints toward Happy (0.80, 0.80):
  → (0.20, 0.20)  — match: meets you where you are
  → (0.50, 0.50)  — bridge: starts moving you
  → (0.80, 0.80)  — destination: where you're headed
   ↓
For each waypoint: search 89,740 songs for the nearest match
   ↓
For each found song: run it through the MLP neural network to predict its emotion
   ↓
Return 3 songs + Spotify links to the browser
```

Everything else in the codebase is either setting up for this moment or displaying the result.

---

## Part 1 — The Backend

The backend lives in `backend/src/`. Six Python files, each with one job.

---

### `label_emotions.py` — Stamp every song with an emotion

**What it does:**
Reads 114,000 raw Spotify songs from `data/raw/dataset.csv`, looks at each song's `valence` and `energy` values, and assigns one of six emotion labels.

**Why this is needed:**
The dataset doesn't come with emotion labels. Songs only have audio features like `valence=0.72, energy=0.65`. We need to turn those numbers into emotion words so the ML models have something to learn.

**How the labeling works:**
Each emotion owns a region of the valence × energy space:

```
valence > 0.6  AND  energy > 0.6          → Happy
valence < 0.4  AND  energy < 0.4          → Sad
valence < 0.4  AND  energy > 0.65         → Angry
valence < 0.5  AND  0.35 ≤ energy ≤ 0.7  AND  minor key  → Fear
valence < 0.35 AND  energy < 0.5          → Disgust
valence ≥ 0.5  AND  energy ≥ 0.65         → Surprise
```

Songs that fall in the "grey zone" (none of the rules match) get assigned to whichever emotion centroid is closest — using simple Euclidean distance on the (valence, energy) plane. No song is discarded.

**Important design decision:**
Previously, Surprise was a catch-all for any song that didn't match the other five rules. This meant 35% of songs were labelled Surprise, flooding the dataset with junk labels. Giving Surprise a real zone (`v ≥ 0.5 AND e ≥ 0.65`) fixed the imbalance.

**Output:** `data/processed/dataset_labeled.csv` — same songs, now with an `emotion` column.

---

### `train_classifier.py` — Build and compare two ML models

**What it does:**
Reads the labeled dataset, trains two classifiers, compares them, and saves the winner (MLP) as a `.pkl` file.

**The two models:**

**Random Forest** — the baseline. 100 decision trees vote on each song's emotion. It learns rules like "if valence > 0.6 and energy > 0.7 → probably Happy". Because its training labels came from these exact rules, it scores 100% accuracy — not a bug, just a known limitation called *label circularity*. It still serves as the baseline comparison.

**MLP Neural Network** — the one that matters. Three layers of numbers:
- Input: 8 audio features (valence, energy, tempo, mode, danceability, loudness, acousticness, genre)
- Hidden layer 1: 128 neurons (ReLU activation)
- Hidden layer 2: 64 neurons (ReLU activation)
- Output: 6 neurons, one per emotion (softmax — highest wins)

The MLP is wrapped in a `Pipeline` with a `StandardScaler` that normalises the inputs before training. It uses `early_stopping=True`, meaning training stops automatically when the model stops improving on a held-out 10% of data — preventing overfitting. Accuracy: 99.26%, converged at 39 iterations.

**Output files:**
- `models/emotion_classifier_rf.pkl` — Random Forest
- `models/emotion_classifier_mlp.pkl` — MLP Neural Network
- `models/genre_label_encoder.pkl` — encodes genre strings as integers (needed for MLP inference)

---

### `mood_parser.py` — Convert text to an emotion label

**What it does:**
Takes one string of user text and calls OpenAI's GPT-4o-mini to classify it into exactly one of six emotions.

**How it works:**
The prompt is strict and explicit. It tells GPT-4o-mini: "You are a music therapist. Classify this into one of: Happy, Sad, Angry, Fear, Disgust, Surprise. Return only the word." The model is called with `temperature=0` so the same input always produces the same output.

```python
"I feel completely hopeless"  →  "Sad"
"I could punch through a wall"  →  "Angry"
```

**Why GPT-4o-mini specifically:**
Cheapest OpenAI model, fast response time (<200ms), and excellent at short classification tasks. A larger model adds no value here.

---

### `recommender.py` — The heart of the system

**What it does:**
Takes an emotion label, finds 3 songs that form an ISO journey toward Happy, and for each song runs live MLP inference to confirm its emotion.

**Step by step:**

1. **Map emotion to coordinates.** Each emotion has a known (valence, energy) position on Russell's Circumplex:
   ```
   Sad     → (0.20, 0.20)
   Happy   → (0.80, 0.80)    ← always the destination
   Angry   → (0.20, 0.80)
   etc.
   ```

2. **Calculate 3 waypoints.** The path from the detected emotion toward Happy:
   ```
   Waypoint 1 (match):       (current_v, current_e)
   Waypoint 2 (bridge):      average of current + (0.80, 0.80)
   Waypoint 3 (destination): (0.80, 0.80)
   ```

3. **Find nearest song.** For each waypoint, search the 89,740-song CSV for the closest match using Euclidean distance on valence and energy. Instead of always picking the single nearest song, it samples randomly from the top-K closest:
   - Waypoint 1: top 30 → random pick
   - Waypoint 2: top 20 → random pick
   - Waypoint 3: top 10 → random pick

   This is why every session gives different songs even if your mood is the same.

4. **MLP live inference.** For each selected song, the MLP neural network runs in real time:
   ```python
   features = [valence, energy, tempo, mode, danceability, loudness, acousticness, genre_encoded]
   predicted_emotion = mlp.predict([features])[0]
   ```
   The emotion label you see in the app is a fresh neural network prediction — not a pre-assigned tag from the CSV.

---

### `pipeline.py` — The CLI conversation loop

**What it does:**
Runs a text-based terminal app. Loads the dataset and models once, then loops forever: ask user for mood → call `mood_parser.py` → call `recommender.py` → print 3 songs → repeat.

This is the demo mode accessed via `make run`. It's the exact same engine as the web app but in a terminal.

---

### `api.py` — The FastAPI REST server

**What it does:**
Exposes the same engine as `pipeline.py` but over HTTP, so the React frontend can call it.

**On startup** it loads:
- The 89,740-song CSV into memory
- The MLP `.pkl` file
- The genre encoder `.pkl` file

These are loaded once and reused for every request — no disk reads on each API call.

**Two endpoints:**

`POST /api/recommend`
```json
Request:  { "text": "I feel really low today" }
Response: { "emotion": "Sad", "songs": [ {...}, {...}, {...} ] }
```

`GET /health`
```json
{ "status": "ok", "songs_loaded": 89740, "model": "MLP loaded" }
```

Pydantic validates every request and response automatically. CORS is enabled for localhost so the React app can call it during development.

---

## Part 2 — The Frontend

The frontend lives in `frontend/src/`. Every file is TypeScript (`.ts` or `.tsx`). No JavaScript.

---

### `types.ts` — The contract

**What it does:**
Defines the exact shape of every data object that flows through the app.

```typescript
type Emotion = 'Happy' | 'Sad' | 'Angry' | 'Fear' | 'Disgust' | 'Surprise'
```

This is a *union type* — TypeScript will refuse to compile if any code tries to pass a string that isn't one of these six words. This catches bugs at compile time, not at runtime.

```typescript
interface Song {
  track_name: string
  artists: string
  emotion: Emotion      // ← strict, not just 'string'
  valence: number
  energy: number
  track_id: string
  spotify_url: string
}
```

Everything else in the frontend imports from this file. It is the single source of truth for what data looks like.

---

### `lib/emotions.ts` — Emotion colours and labels

**What it does:**
A lookup table that maps each emotion to its visual identity.

```typescript
Happy:    { color: '#d97706', ... }   // warm amber
Sad:      { color: '#2563eb', ... }   // blue
Angry:    { color: '#dc2626', ... }   // red
Fear:     { color: '#7c3aed', ... }   // purple
Disgust:  { color: '#059669', ... }   // green
Surprise: { color: '#ea580c', ... }   // orange
```

Two utility functions:
- `emotionColor(emotion)` → returns the hex colour for that emotion
- `emotionBg(emotion)` → returns a very light version for backgrounds

The entire UI's colour scheme is driven from this one file. When the emotion changes, all tints, borders, and indicators update automatically.

Also stores `STEP_META` — the three labels used on song cards: "matching your mood", "the bridge", "your destination".

---

### `hooks/useRecommend.ts` — The API layer

**What it does:**
A custom React hook that handles all communication with the backend.

**Why it's a hook (not just a function):**
React components re-render when data changes. A hook lets you wire up loading states, results, and errors directly to React's rendering system — so the UI updates automatically when the API call completes.

**What it manages:**
- `result` — the 3 songs (or null if no request yet)
- `loading` — true while waiting for the API
- `error` — error message (or null if fine)
- `recommend(text)` — makes the POST request, returns `true` on success or `false` on failure
- `reset()` — clears everything back to starting state

**The `Promise<boolean>` return value is intentional:**
`MoodInput` uses it: if `recommend()` returns `true`, the textarea clears automatically. If it returns `false` (API error), the text stays so you don't have to retype it.

**`useCallback` wraps the functions:**
This prevents React from recreating these functions on every render, which would cause unnecessary re-renders downstream. A small but correct optimisation.

---

### `components/MoodInput.tsx` — The text input form

**What it does:**
The textarea where you describe your mood, plus the submit button.

**Key behaviours:**
- Auto-focuses on page load (`useEffect` with `textareaRef.current?.focus()`)
- The textarea border glows in the emotion's accent colour as you type
- Enter submits the form; Shift+Enter adds a new line
- While loading: button switches from "Find my path" to "Reading your mood…" with a spinning icon, animated with `framer-motion`'s `AnimatePresence`
- On success: clears the textarea (because `recommend()` returned `true`)
- Button changes colour and disables when there's no text

**Why manage hover colour with JS here:**
The submit button background needs to be the emotion's accent colour, which is dynamic. Tailwind CSS can't do dynamic colours at runtime — it only knows classes defined at build time. So inline `style` props are the correct approach for runtime-dynamic colours.

---

### `components/ResultsPanel.tsx` — The 3-song results view

**What it does:**
Displays the results after the API responds. Shows the detected emotion badge, the ISO principle description, and three song cards.

**Layout decision:**
The emotion badge sits right-aligned on the same row as the ISO description. This was a deliberate choice: the badge is secondary information (you already typed your mood), so it doesn't need its own prominent row.

**Reset button:**
"Try a different mood" clears all results and takes you back to the input. The icon rotates 180° on hover — a subtle affordance that it's a refresh action.

---

### `components/SongCard.tsx` — One song's display

**What it does:**
Renders a single song with its title, artist, emotion badge, valence/energy readout, Spotify button, and ResonanceMap.

**The left border:**
Songs 1 and 2 (match and bridge) get a coloured left border in the detected emotion's accent colour. Song 3 (destination) always gets gold (`#d97706`) — marking it as the target.

**The emotion pill on each card:**
Each song's emotion is the MLP's live prediction. It might differ from the detected mood — for example, the bridge song might be labelled "Fear" because it's in a minor key but moving upward in energy. This is correct and expected.

**Hover on the Spotify button:**
Uses `useState<boolean>` to track hover state and drives the background colour from that state variable. This is the React-correct pattern — direct DOM mutation (`e.currentTarget.style.backgroundColor`) bypasses React's rendering model and causes subtle bugs. Using state keeps everything predictable.

**Animation:**
Each card fades in and slides up slightly with a staggered delay (index × 180ms) using `framer-motion`. The three cards arrive sequentially, not all at once.

---

### `components/ResonanceMap.tsx` — The signature visual element

**What it does:**
A 72×72 SVG diagram showing exactly where a song sits on Russell's Circumplex Model. X-axis = valence, Y-axis = energy. A glowing dot marks the song's position.

**Why it exists:**
This is the academic backbone made visible. You can watch the dot migrate across the three cards — starting in the bottom-left for a sad match, moving toward the top-right for the happy destination. It makes the ISO journey tangible.

**The subtle warm circle in the top-right:**
A faint amber indicator marks the "Happy" target zone (valence=0.80, energy=0.80) on every map. Even for the matching song (bottom-left), you can see where it's heading.

**The `useId()` bug fix:**
SVG filters (the glow effect on the dot) are referenced by ID. If two cards on the same page happen to have the same filter ID, they silently share the same filter — and one of them loses its glow. React 18's `useId()` hook generates a guaranteed-unique ID per component instance. Every `ResonanceMap` now has its own private filter ID. Without this fix, the bug would appear unpredictably whenever two songs landed at similar coordinates.

---

### `App.tsx` — The orchestrator

**What it does:**
The top-level component that holds everything together and manages global state.

**Emotion-reactive background:**
When the API returns a result, the entire page background gradient shifts to a very light tint of the detected emotion's colour. The top-1px accent stripe also changes. Both transitions use `duration-700` (700ms fade) so the shift is noticeable but not jarring.

**Auto-scroll:**
A `useEffect` watches `result`. When it changes from null to a real result, it smoothly scrolls the results panel into view. Without this, the results would render below the fold and you'd have to manually scroll.

**State ownership:**
All state (`result`, `loading`, `error`) lives in `App.tsx` via the `useRecommend` hook. Components receive what they need as props. No prop-drilling problem here because the tree is shallow.

---

## Part 3 — How Frontend and Backend Connect

```
User types text
       ↓
MoodInput.tsx calls onSubmit(text)
       ↓
App.tsx receives it, passes to useRecommend hook
       ↓
useRecommend.ts: POST http://localhost:8000/api/recommend
  Body: { "text": "I feel really low today" }
       ↓
FastAPI (api.py) receives it
  → mood_parser.py → "Sad"
  → recommender.py → 3 songs with MLP predictions
  → returns JSON
       ↓
useRecommend.ts: setResult(data), returns true
       ↓
MoodInput clears its textarea
App.tsx renders ResultsPanel
ResultsPanel renders 3 × SongCard
Each SongCard renders a ResonanceMap
Page background tints to Sad's blue
```

---

## Part 4 — What Changed in the Backend

Before our work, the backend had three significant problems:

### Problem 1: The MLP model was never used

The `.pkl` model file was trained and saved, but `recommender.py` never loaded it. Recommendations were based solely on pre-assigned CSV labels — the neural network was trained and immediately ignored.

**Fix:** Added a `_mlp_predict()` function in `recommender.py` that loads the MLP and genre encoder, and calls `mlp.predict()` on each selected song's 8 features. Now the model does real work.

### Problem 2: The same song appeared every time

The song nearest to the "Happy" destination (valence=0.80, energy=0.80) was `Vai Vadiar` with a distance of 0.002. It was always Song 3, every session.

**Fix:** Changed from "always pick nearest" to "sample randomly from top-K nearest". The destination now picks from the 10 closest songs — so you get variety while still staying within the right emotional zone.

### Problem 3: Surprise had 35% of all songs

Surprise was the catch-all: any song that didn't match the other five rules became Surprise. This meant a happy upbeat song with valence=0.9 would become Surprise if it had slightly lower energy than 0.8. It ruined the bridge zone search.

**Fix:** Gave Surprise a real zone (`valence ≥ 0.5 AND energy ≥ 0.65`). Mid-zone songs now go to the nearest emotion centroid in the valence-energy space. Result: Surprise dropped from 35% → ~15% of the dataset, and the distribution became meaningful.

---

## Part 5 — The Academic Concepts Made Simple

### ISO Principle
A technique from music therapy. You can't jump from sad to happy — the gap creates emotional dissonance (like someone blasting club music at a funeral). Instead: first play music that matches the current mood (meet the person where they are), then gradually shift, then arrive at the target. Three songs, three steps.

### Russell's Circumplex Model (1980)
A 2D map of human emotions. Instead of just naming emotions, Russell showed they live in a 2D space with two axes:
- Valence: how positive or negative you feel
- Arousal (energy): how activated or calm you feel

Any emotion can be plotted as an (x, y) coordinate. This is why we can do maths on emotions — calculate midpoints, measure distance, plot trajectories.

### Ekman's 6 Basic Emotions
Paul Ekman's research identified emotions that appear universally across cultures. We use six: Happy, Sad, Angry, Fear, Disgust, Surprise. These are the only six labels in our system.

### Label Circularity
Why is the Random Forest 100% accurate? Because we created the labels with rules (`valence > 0.6 AND energy > 0.6 → Happy`), then trained the Random Forest on those exact same features. The model is learning the rules back from the labels they generated — of course it's 100%. This is a known limitation, acknowledged in the defense.

---

## Defense Quick Reference

**"What is the ISO Principle?"**
Match the user's current mood first, then bridge, then guide to the destination. You can't skip directly to happy.

**"What neural network did you build?"**
An MLP with two hidden layers: 128 → 64 neurons, ReLU activations, 8 input features, 6 output classes. It runs live on every recommendation request — the emotion labels in the app are real-time predictions, not pre-assigned tags.

**"Why is your Random Forest 100% accurate?"**
Label circularity — the labels were created from the same features the RF trains on. It's acknowledged as a known limitation. The MLP's 99.26% comes from a stratified 80/20 train-test split with proper early stopping.

**"What are Spotify links and how do they work?"**
The dataset includes Spotify track IDs. The URL `https://open.spotify.com/track/{id}` works publicly for any track. No API key, no authentication — the link just works.

**"How does the emotion colour change in the UI?"**
The detected emotion (from GPT-4o-mini) drives a colour lookup in `lib/emotions.ts`. That colour propagates: top accent stripe, page background gradient, card borders, input glow. All via React state — no direct DOM manipulation.
