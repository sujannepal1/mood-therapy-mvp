"""
recommendation/mood_parser.py

Sends user mood text to OpenAI GPT-4o-mini with a strict classification prompt.
Returns exactly one of 6 emotion labels: Happy, Sad, Angry, Fear, Disgust, Surprise.

Cost: ~$0.0001 per call (10 output tokens max at GPT-4o-mini pricing).
"""

import os
import time

from loguru import logger

# from openai import OpenAI
from dotenv import load_dotenv

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


# def parse_mood(user_text: str) -> str:
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#     logger.info(
#         f"[MOOD] parsing: '{user_text[:80]}{'...' if len(user_text) > 80 else ''}'"
#     )

#     t0 = time.perf_counter()
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         max_tokens=10,
#         temperature=0,
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": user_text},
#         ],
#     )
#     elapsed_ms = (time.perf_counter() - t0) * 1000

#     raw = response.choices[0].message.content.strip()
#     label = raw.capitalize()

#     if label not in VALID_EMOTIONS:
#         logger.warning(f"[MOOD] unexpected label from GPT: {raw!r} — defaulting to Sad")
#         return "Sad"

#     logger.info(f"[MOOD] '{label}' ({elapsed_ms:.0f}ms)")
#     return label


def get_mock_emotion(text: str) -> str:
    # Mock function to simulate mood parsing for testing
    text = text.lower()
    mood_mapping = {
        "happy": "Happy",
        "sad": "Sad",
        "angry": "Angry",
        "fear": "Fear",
        "disgust": "Disgust",
        "surprise": "Surprise",
    }
    return mood_mapping.get(text, "Happy")  # Default to "Happy" if not found
