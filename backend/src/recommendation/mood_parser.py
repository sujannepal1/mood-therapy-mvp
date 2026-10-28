"""
recommendation/mood_parser.py

Sends user mood text to OpenAI GPT-4o-mini with a strict classification prompt.
Returns exactly one of 6 emotion labels: Happy, Sad, Angry, Fear, Disgust, Surprise.

If OPENAI_API_KEY is not set (or the call fails), falls back to a local
keyword classifier so the app still runs without an API key.

Cost: ~$0.0001 per call (10 output tokens max at GPT-4o-mini pricing).
"""

import os
import re
import time

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

VALID_EMOTIONS = {"Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise"}

SYSTEM_PROMPT = """You are an emotion classifier for a music therapy application.
The user will describe how they are feeling in natural language.
Classify their mood into exactly ONE of these 6 labels:
Happy, Sad, Angry, Fear, Disgust, Surprise

Rules:
- Return ONLY the label. No punctuation. No explanation. Nothing else.
- If the mood is ambiguous, return the closest match.
- Never return anything other than one of the 6 labels above."""

KEYWORDS: dict[str, list[str]] = {
    "Sad": [
        "sad", "empty", "hopeless", "lonely", "alone", "depress", "down", "cry",
        "crying", "tears", "heartbroken", "broken", "miss", "grief", "lost",
        "tired", "exhausted", "numb", "worthless", "blue", "unhappy", "low",
        "dukhi", "udas",
    ],
    "Angry": [
        "angry", "anger", "mad", "furious", "rage", "hate", "annoyed", "irritat",
        "frustrat", "pissed", "livid", "resent", "fed up", "rissa",
    ],
    "Fear": [
        "fear", "afraid", "scared", "anxious", "anxiety", "nervous", "worried",
        "worry", "panic", "terrified", "stress", "overwhelm", "uneasy", "dread",
        "dar",
    ],
    "Disgust": [
        "disgust", "gross", "sick of", "revolt", "repuls", "nause", "ashamed",
        "shame", "yuck", "vile", "contempt",
    ],
    "Surprise": [
        "surprise", "surprised", "shocked", "shock", "amazed", "astonish",
        "unexpected", "wow", "can't believe", "cannot believe", "stunned",
    ],
    "Happy": [
        "happy", "joy", "glad", "great", "good", "excited", "love", "awesome",
        "amazing", "cheerful", "content", "grateful", "blessed", "wonderful",
        "fantastic", "relaxed", "calm", "khusi",
    ],
}

NEGATIONS = ("not ", "n't ", "never ", "no ")


def classify_by_keywords(user_text: str) -> str:
    text = f" {user_text.lower()} "
    scores = {emotion: 0 for emotion in KEYWORDS}

    for emotion, words in KEYWORDS.items():
        for word in words:
            for match in re.finditer(rf"\b{re.escape(word)}", text):
                preceding = text[max(0, match.start() - 12) : match.start()]
                negated = any(neg in preceding for neg in NEGATIONS)
                if negated and emotion == "Happy":
                    scores["Sad"] += 1
                elif not negated:
                    scores[emotion] += 1

    best = max(scores, key=lambda e: scores[e])
    return best if scores[best] > 0 else "Sad"


def _parse_with_openai(user_text: str, api_key: str) -> str | None:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=10,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    raw = (response.choices[0].message.content or "").strip()
    label = raw.capitalize()

    if label not in VALID_EMOTIONS:
        logger.warning(f"[MOOD] unexpected label from GPT: {raw!r}")
        return None

    logger.info(f"[MOOD] GPT → '{label}' ({elapsed_ms:.0f}ms)")
    return label


def parse_mood(user_text: str) -> str:
    logger.info(
        f"[MOOD] parsing: '{user_text[:80]}{'...' if len(user_text) > 80 else ''}'"
    )

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        try:
            label = _parse_with_openai(user_text, api_key)
            if label:
                return label
        except Exception as exc:
            logger.warning(f"[MOOD] OpenAI call failed ({type(exc).__name__}) — using keyword fallback")

    label = classify_by_keywords(user_text)
    logger.info(f"[MOOD] keywords → '{label}'")
    return label
