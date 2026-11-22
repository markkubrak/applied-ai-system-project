"""
AI-powered music recommender using Retrieval-Augmented Generation (RAG).

Pipeline for a natural-language request such as "calm music for studying, no vocals":

  1. PARSE     - Gemini turns the request into a structured preference dict.
  2. GUARDRAIL - validate_prefs() clamps numbers, checks genre/mood against the
                 catalog, and fills defaults, so bad LLM output cannot break scoring.
  3. RETRIEVE  - the existing scoring engine (recommend_songs) ranks the catalog
                 and returns the top candidates with score explanations.
  4. GENERATE  - Gemini writes the recommendation using ONLY the retrieved songs
                 and their score breakdowns as context.
  5. VERIFY    - every song the LLM names is checked against the retrieved set;
                 invented songs are dropped. If the AI fails at any step, the
                 system falls back to the plain ranked list.

Every step is logged to logs/recommender.log.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol

from src.recommender import recommend_songs

LOG_DIR = Path("logs")
DEFAULT_MODEL = "gemini-3.8-flash"
NUMERIC_KEYS = ["target_energy", "target_acousticness", "target_valence",
                "target_instrumentalness", "discovery"]
DEFAULT_PREFS = {
    "favorite_genre": "",
    "favorite_mood": "",
    "target_energy": 0.5,
    "target_acousticness": 0.5,
    "target_valence": 0.5,
    "target_instrumentalness": 0.5,
    "discovery": 0.5,
}
MAX_REQUEST_CHARS = 500
MAX_ATTEMPTS = 2
TIMEOUT_MS = 25_000
RETRYABLE = ("429", "500", "503", "504", "UNAVAILABLE", "DEADLINE_EXCEEDED", "RESOURCE_EXHAUSTED", "timed out")


def get_logger() -> logging.Logger:
    """Return the shared logger, writing to logs/recommender.log (created on first use)."""
    logger = logging.getLogger("music_ai")
    if not logger.handlers:
        LOG_DIR.mkdir(exist_ok=True)
        handler = logging.FileHandler(LOG_DIR / "recommender.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


log = get_logger()


class LLMClient(Protocol):
    """Anything with a generate(prompt) -> str method (real Gemini or a test fake)."""
    def generate(self, prompt: str) -> str: ...


class GeminiClient:
    """Thin wrapper around the Google Gen AI SDK."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (see README setup steps).")
        from google import genai  # imported here so the non-AI code works without the SDK
        # Timeout (ms) so a slow API call fails over to the fallback instead of hanging
        self.client = genai.Client(api_key=api_key, http_options={"timeout": TIMEOUT_MS})
        self.on_status: Optional[Callable[[str], None]] = None  # set by recommend_from_text
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)

    def generate(self, prompt: str) -> str:
        """Call Gemini, retrying temporary errors (overload / rate limit) with backoff."""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "temperature": 0.2,
                        # Short structured tasks don't need long reasoning; keeps calls fast
                        "thinking_config": {"thinking_level": "low"},
                    },
                )
                return response.text or ""
            except Exception as exc:
                temporary = any(code in str(exc) for code in RETRYABLE)
                if not temporary or attempt == MAX_ATTEMPTS:
                    raise
                wait = 2 ** attempt
                log.warning("Gemini attempt %d failed (%s); retrying in %ds",
                            attempt, str(exc)[:80], wait)
                if self.on_status:
                    self.on_status(f"Gemini is busy, retrying in {wait}s...")
                time.sleep(wait)


def describe_error(exc: Exception) -> str:
    """Short, user-friendly error text (the full error still goes to the log)."""
    if any(code in str(exc) for code in RETRYABLE):
        return "Gemini is overloaded or rate-limited right now; try again in a few minutes"
    return str(exc)


def extract_json(text: str) -> Dict:
    """Parse a JSON object from LLM output, tolerating ```json fences or extra text."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in LLM response")
    return json.loads(match.group(0))


# ---------- Step 1 + 2: parse the request, then validate ----------

def build_parse_prompt(request: str, genres: List[str], moods: List[str]) -> str:
    return f"""You convert a music request into listening preferences.
Request: "{request}"

Return ONLY a JSON object with these keys:
  favorite_genre: one of {genres} or "" if none fits
  favorite_mood: one of {moods} or "" if none fits
  target_energy: 0.0 (very calm) to 1.0 (very intense)
  target_acousticness: 0.0 (electronic/produced) to 1.0 (acoustic)
  target_valence: 0.0 (sad/dark) to 1.0 (happy/bright)
  target_instrumentalness: 0.0 (wants vocals) to 1.0 (no vocals)
  discovery: 0.0 (popular hits) to 1.0 (obscure hidden gems), 0.5 if not mentioned
"""


def validate_prefs(raw: Dict, genres: List[str], moods: List[str]) -> tuple:
    """Guardrail: coerce LLM output into safe prefs. Returns (prefs, warnings)."""
    prefs = dict(DEFAULT_PREFS)
    warnings = []

    for key, allowed in (("favorite_genre", genres), ("favorite_mood", moods)):
        value = str(raw.get(key, "")).strip().lower()
        if value and value not in allowed:
            warnings.append(f"{key} '{value}' is not in the catalog; ignored")
            value = ""
        prefs[key] = value

    for key in NUMERIC_KEYS:
        try:
            value = float(raw.get(key, DEFAULT_PREFS[key]))
        except (TypeError, ValueError):
            warnings.append(f"{key} was not a number; using default")
            continue
        if not 0.0 <= value <= 1.0:
            warnings.append(f"{key} {value} out of range; clamped")
            value = min(1.0, max(0.0, value))
        prefs[key] = value

    return prefs, warnings


def parse_request(request: str, songs: List[Dict], client: LLMClient) -> tuple:
    """Use the LLM to turn free text into validated prefs. Returns (prefs, warnings)."""
    genres = sorted({s["genre"] for s in songs})
    moods = sorted({s["mood"] for s in songs})
    raw = extract_json(client.generate(build_parse_prompt(request, genres, moods)))
    log.info("parsed raw prefs: %s", raw)
    return validate_prefs(raw, genres, moods)


# ---------- Step 4 + 5: grounded generation, then verification ----------

def build_answer_prompt(request: str, retrieved: List[tuple]) -> str:
    context = "\n".join(
        f"- \"{song['title']}\" by {song['artist']} ({song['genre']}, {song['mood']}), "
        f"score {score:.2f}. Why it matched: {why}"
        for song, score, why in retrieved
    )
    return f"""You are a music recommendation assistant.
A listener asked: "{request}"

Our catalog search retrieved these candidate songs, best match first:
{context}

Using ONLY the songs listed above (never invent songs), pick up to 3 that best fit
the request. Base each reason on the match details given. Return ONLY JSON:
{{"summary": "one friendly sentence", "picks": [{{"title": "exact title", "reason": "one sentence"}}]}}
"""


def verify_picks(answer: Dict, retrieved: List[tuple]) -> tuple:
    """Guardrail: keep only picks whose title is in the retrieved set. Returns (picks, warnings)."""
    by_title = {song["title"].lower(): song for song, _, _ in retrieved}
    picks, warnings = [], []
    for pick in answer.get("picks", []):
        title = str(pick.get("title", "")).strip()
        song = by_title.get(title.lower())
        if song is None:
            warnings.append(f"LLM suggested '{title}', which was not retrieved; dropped")
            continue
        picks.append({"song": song, "reason": str(pick.get("reason", "")).strip()})
    return picks, warnings


def fallback_picks(retrieved: List[tuple], n: int = 3) -> List[Dict]:
    """Non-AI answer: the top retrieved songs with their score explanations."""
    return [{"song": song, "reason": why} for song, _, why in retrieved[:n]]


# ---------- Full pipeline ----------

def recommend_from_text(request: str, songs: List[Dict],
                        client: Optional[LLMClient] = None, k: int = 5,
                        on_status: Optional[Callable[[str], None]] = None) -> Dict:
    """
    Run the full RAG pipeline. Never raises for AI problems; instead returns a
    result with used_ai=False and warnings explaining what went wrong.
    """
    request = (request or "").strip()
    if not request:
        raise ValueError("request must not be empty")
    if len(request) > MAX_REQUEST_CHARS:
        log.warning("request truncated from %d chars", len(request))
        request = request[:MAX_REQUEST_CHARS]

    status = on_status or (lambda msg: None)
    log.info("request: %r", request)
    result = {"request": request, "prefs": dict(DEFAULT_PREFS), "retrieved": [],
              "summary": "", "picks": [], "used_ai": False, "warnings": []}

    if client is None:
        try:
            client = GeminiClient()
        except Exception as exc:  # missing key or SDK
            result["warnings"].append(f"AI unavailable ({exc}); using default preferences")
            log.warning("AI unavailable: %s", exc)

    if client is not None and hasattr(client, "on_status"):
        client.on_status = on_status

    # Steps 1-2: parse + validate
    if client is not None:
        status("Understanding your request...")
        try:
            prefs, warns = parse_request(request, songs, client)
            result["prefs"] = prefs
            result["warnings"] += warns
        except Exception as exc:
            result["warnings"].append(f"could not understand request ({describe_error(exc)}); using defaults")
            log.error("parse failed: %s", exc)
    log.info("validated prefs: %s", result["prefs"])

    # Step 3: retrieve
    status("Searching the catalog...")
    retrieved = recommend_songs(result["prefs"], songs, k)
    result["retrieved"] = retrieved
    log.info("retrieved: %s", [s["title"] for s, _, _ in retrieved])

    # Steps 4-5: generate from retrieved context, then verify
    if client is not None:
        status("Writing recommendations...")
        try:
            answer = extract_json(client.generate(build_answer_prompt(request, retrieved)))
            picks, warns = verify_picks(answer, retrieved)
            result["warnings"] += warns
            if picks:
                result.update(summary=str(answer.get("summary", "")).strip(),
                              picks=picks, used_ai=True)
            else:
                result["warnings"].append("LLM returned no valid picks; using ranked list")
        except Exception as exc:
            result["warnings"].append(f"could not generate answer ({describe_error(exc)}); using ranked list")
            log.error("generation failed: %s", exc)

    if not result["picks"]:
        result["summary"] = "Here are the closest matches from the catalog."
        result["picks"] = fallback_picks(retrieved)

    for w in result["warnings"]:
        log.warning(w)
    log.info("final picks (used_ai=%s): %s", result["used_ai"],
             [p["song"]["title"] for p in result["picks"]])
    return result
