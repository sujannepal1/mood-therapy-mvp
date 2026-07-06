"""
common/features.py

Shared librosa audio feature extraction used by both:
  - collect_spotify_data.py  (building the Nepali dataset)
  - classification/classifier.py  (classifying an uploaded song)

Returns the 7-feature dict the Nepali MLP was trained on:
  valence, energy, tempo, mode, loudness, acousticness, danceability
"""

import numpy as np

# Krumhansl-Schmuckler key profiles for major/minor detection
_KS_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
_KS_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])


def extract_features(audio_path: str) -> dict | None:
    """
    Load an audio file and compute the 7 features the Nepali MLP uses.

    Returns None if the file cannot be loaded or processed.
    """
    try:
        import librosa

        y, sr = librosa.load(audio_path, mono=True)
    except Exception:
        return None

    try:
        # ── Energy ────────────────────────────────────────────────────────────
        rms    = librosa.feature.rms(y=y)
        energy = float(np.clip(np.mean(rms) / 0.15, 0, 1))

        # ── Tempo ──────────────────────────────────────────────────────────────
        tempo_raw, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo        = float(np.atleast_1d(tempo_raw)[0])
        tempo_norm   = float(np.clip((tempo - 60) / 120, 0, 1))

        # ── Mode (Krumhansl-Schmuckler) ────────────────────────────────────────
        chroma      = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        major_corr  = max(float(np.corrcoef(np.roll(_KS_MAJOR, i), chroma_mean)[0, 1]) for i in range(12))
        minor_corr  = max(float(np.corrcoef(np.roll(_KS_MINOR, i), chroma_mean)[0, 1]) for i in range(12))
        mode        = 1 if major_corr >= minor_corr else 0

        # ── Loudness ───────────────────────────────────────────────────────────
        loudness = float(librosa.amplitude_to_db(np.array([float(np.mean(np.abs(y)))]))[0])

        # ── Acousticness (harmonic energy ratio) ───────────────────────────────
        harmonic, percussive = librosa.effects.hpss(y)
        h_e          = float(np.mean(harmonic ** 2))
        p_e          = float(np.mean(percussive ** 2))
        acousticness = float(h_e / (h_e + p_e + 1e-10))

        # ── Danceability (onset autocorrelation) ───────────────────────────────
        onset_env    = librosa.onset.onset_strength(y=y, sr=sr)
        ac           = librosa.util.normalize(librosa.autocorrelate(onset_env))
        danceability = float(np.clip(float(np.max(ac[2:])) * 1.2, 0, 1))

        # ── Valence (psychoacoustic approximation) ─────────────────────────────
        # major key → positive; faster tempo → positive; brighter timbre → positive
        centroid      = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_norm = float(np.clip(float(np.mean(centroid)) / 4000, 0, 1))
        valence       = float(np.clip(0.40 * mode + 0.35 * tempo_norm + 0.25 * centroid_norm, 0.05, 0.95))

        return {
            "valence":      round(valence,      3),
            "energy":       round(energy,       3),
            "tempo":        round(tempo,        1),
            "mode":         mode,
            "loudness":     round(loudness,     2),
            "acousticness": round(acousticness, 3),
            "danceability": round(danceability, 3),
        }

    except Exception:
        return None
