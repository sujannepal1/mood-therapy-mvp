"""
collect_spotify_data.py

Pulls Nepali emotion playlists from Spotify, downloads each song via yt-dlp,
extracts audio features with librosa, and saves a labeled CSV ready for training.

Uses Client Credentials auth — no browser login needed, just CLIENT_ID + CLIENT_SECRET.
Playlists must be PUBLIC on Spotify for this to work.

Run all six playlists:
  cd backend
  uv run python src/collect_spotify_data.py

Run one playlist only:
  uv run python src/collect_spotify_data.py Sad

Test with first 5 songs:
  uv run python src/collect_spotify_data.py Happy --limit 5
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import spotipy
from dotenv import load_dotenv
from loguru import logger
from spotipy.oauth2 import SpotifyClientCredentials

from common.features import extract_features as _extract_from_file

load_dotenv()

OUTPUT_PATH = Path(__file__).parent.parent / "data/processed/nepali_dataset.csv"

# Hardcoded public playlist IDs — update here if playlists change
EMOTION_PLAYLISTS = {
    "Happy":    "4NOiv5VZilzp3VGrFtmReS",
    "Sad":      "2hheZ7P3hLFbkKPURCiZL4",
    "Angry":    "0s2cQ9t6YHoYt0kO8LkhK7",
    "Fear":     "4yhfOLud5ZqT1Zmaoe4P3c",
    "Disgust":  "1V04P09w7AobPtKicvMXwi",
    "Surprise": "0fJxnP7fkoYyVP9uUcvAA1",
}


def get_client() -> spotipy.Spotify:
    return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
    ))


def get_playlist_tracks(sp: spotipy.Spotify, playlist_id: str, limit: int | None = None) -> list[dict]:
    tracks: list[dict] = []
    results = sp.playlist_tracks(playlist_id)

    while results:
        for item in results["items"]:
            track = item.get("track")
            if not track or not track.get("id"):
                continue
            tracks.append({
                "id":      track["id"],
                "name":    track["name"],
                "artists": ", ".join(a["name"] for a in track.get("artists", [])),
            })
            if limit and len(tracks) >= limit:
                return tracks
        results = sp.next(results) if results.get("next") else None

    return tracks


def download_and_extract(track_name: str, artists: str) -> dict | None:
    query = f"{track_name} {artists}"
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            subprocess.run([
                "yt-dlp", f"ytsearch1:{query}",
                "--extract-audio", "--audio-format", "mp3", "--audio-quality", "9",
                "--no-playlist", "--quiet",
                "-o", str(Path(tmp_dir) / "%(title)s.%(ext)s"),
            ], capture_output=True, text=True, timeout=90)

            mp3_files = list(Path(tmp_dir).glob("*.mp3"))
            if not mp3_files:
                logger.warning(f"  yt-dlp found nothing for: {query}")
                return None

            return _extract_from_file(str(mp3_files[0]))

    except Exception as e:
        logger.warning(f"  Download/extract failed: {e}")
        return None


def collect(emotions: list[str] | None = None, limit: int | None = None) -> pd.DataFrame:
    sp = get_client()

    # Load already-processed track IDs — only download new songs
    already_done: set[str] = set()
    if OUTPUT_PATH.exists():
        already_done = set(pd.read_csv(OUTPUT_PATH)["track_id"].astype(str))
        logger.info(f"Existing dataset: {len(already_done)} songs — skipping those")

    target_emotions = emotions or list(EMOTION_PLAYLISTS.keys())
    rows: list[dict] = []

    for emotion in target_emotions:
        playlist_id = EMOTION_PLAYLISTS[emotion]
        logger.info(f"\n[{emotion}] Fetching playlist {playlist_id}…")

        tracks    = get_playlist_tracks(sp, playlist_id, limit=limit)
        new_tracks = [t for t in tracks if t["id"] not in already_done]
        skipped   = len(tracks) - len(new_tracks)

        if skipped:
            logger.info(f"[{emotion}] {len(tracks)} in playlist — {skipped} already done, {len(new_tracks)} new")
        else:
            logger.info(f"[{emotion}] {len(new_tracks)} new tracks to process")

        if not new_tracks:
            continue

        for i, track in enumerate(new_tracks):
            logger.info(f"  [{i+1}/{len(new_tracks)}] '{track['name']}' by {track['artists']}")
            features = download_and_extract(track["name"], track["artists"])

            if not features:
                logger.warning(f"  └─ skipped (feature extraction failed)")
                continue

            logger.info(
                f"  └─ V={features['valence']} E={features['energy']} "
                f"T={features['tempo']} mode={'major' if features['mode'] else 'minor'}"
            )

            rows.append({
                "track_id":   track["id"],
                "track_name": track["name"],
                "artists":    track["artists"],
                "emotion":    emotion,
                **features,
            })

            time.sleep(0.05)

    if not rows:
        logger.info("No new songs found across all playlists.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates(subset="track_id")
    logger.info(f"\n{'─'*50}")
    logger.info(f"New songs collected: {len(df)}")
    logger.info(f"\n{df['emotion'].value_counts().to_string()}")
    return df


if __name__ == "__main__":
    emotions = None
    limit    = None

    args = sys.argv[1:]
    if args and not args[0].startswith("--"):
        emotions = [args[0]]
        args = args[1:]
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])

    df = collect(emotions=emotions, limit=limit)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        logger.info("Nothing new to save.")
        sys.exit(0)

    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(subset="track_id")
        logger.info(f"Merged → {len(df)} total tracks")

    df.to_csv(OUTPUT_PATH, index=False)
    logger.success(f"Saved → {OUTPUT_PATH}  ({len(df)} tracks)")
