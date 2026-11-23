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
                # A daily quota won't recover in seconds, so retrying only wastes time
                temporary = any(code in str(exc) for code in RETRYABLE) and not is_daily_quota(exc)
                if not temporary or attempt == MAX_ATTEMPTS:
                    raise
                wait = 2 ** attempt
                log.warning("Gemini attempt %d failed (%s); retrying in %ds",
                            attempt, str(exc)[:80], wait)
                if self.on_status:
                    self.on_status(f"Gemini is busy, retrying in {wait}s...")
                time.sleep(wait)


def is_daily_quota(exc: Exception) -> bool:
    """True when the free-tier daily request limit is used up (resets about once a day)."""
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text and "per day" in text


def describe_error(exc: Exception) -> str:
    """Short, user-friendly error text (the full error still goes to the log)."""
    if is_daily_quota(exc):
        return "the Gemini free-tier daily request limit is used up; it resets within 24 hours"
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


# ---------- Song feature estimation (for songs the user adds) ----------

SONG_RANGES = {  # field -> (min, max, default)
    "energy": (0.0, 1.0, 0.5), "valence": (0.0, 1.0, 0.5), "danceability": (0.0, 1.0, 0.5),
    "acousticness": (0.0, 1.0, 0.5), "instrumentalness": (0.0, 1.0, 0.2),
    "tempo_bpm": (40, 220, 110), "popularity": (0, 100, 50), "release_year": (1900, 2026, 2020),
}
INT_SONG_FIELDS = {"tempo_bpm", "popularity", "release_year"}


def default_song(title: str, artist: str) -> Dict:
    song = {"title": title, "artist": artist, "genre": "", "mood": "", "estimated": False}
    song.update({f: d for f, (_, _, d) in SONG_RANGES.items()})
    return song


def build_estimate_prompt(title: str, artist: str, genres: List[str], moods: List[str]) -> str:
    return f"""Estimate audio features for the song "{title}" by {artist}.
Return ONLY a JSON object with these keys:
  known: true if you recognize this exact song, false if you are guessing
  genre: short lowercase genre; prefer one of {genres} if it fits
  mood: one lowercase word; prefer one of {moods} if it fits
  energy, valence, danceability, acousticness, instrumentalness: numbers 0.0 to 1.0
  tempo_bpm: beats per minute (40-220)
  popularity: 0 (obscure) to 100 (huge hit)
  release_year: four-digit year
"""


def validate_song(raw: Dict, title: str, artist: str) -> tuple:
    """Guardrail: coerce estimated features into valid ranges. Returns (song, warnings)."""
    song, warnings = default_song(title, artist), []
    for key in ("genre", "mood"):
        value = str(raw.get(key, "")).strip().lower()[:30]
        song[key] = value
        if not value:
            warnings.append(f"no {key} estimated; please choose one")
    for field, (low, high, default) in SONG_RANGES.items():
        try:
            value = float(raw.get(field, default))
        except (TypeError, ValueError):
            warnings.append(f"{field} was not a number; using default")
            continue
        if not low <= value <= high:
            warnings.append(f"{field} {value} out of range; clamped")
            value = min(high, max(low, value))
        song[field] = int(round(value)) if field in INT_SONG_FIELDS else round(value, 2)
    song["estimated"] = True
    if raw.get("known") is not True:
        warnings.append("Gemini doesn't recognize this song; values are guesses, please check them")
    return song, warnings


def estimate_song_features(title: str, artist: str, songs: List[Dict],
                           client: Optional[LLMClient] = None) -> tuple:
    """Ask the LLM to estimate a song's features. Never raises for AI problems."""
    title, artist = title.strip(), artist.strip()
    if not title or not artist:
        raise ValueError("title and artist are required")
    try:
        client = client or GeminiClient()
        genres = sorted({s["genre"] for s in songs})
        moods = sorted({s["mood"] for s in songs})
        raw = extract_json(client.generate(build_estimate_prompt(title, artist, genres, moods)))
        log.info("estimated features for %s by %s: %s", title, artist, raw)
        return validate_song(raw, title, artist)
    except Exception as exc:
        log.error("feature estimation failed for %s: %s", title, exc)
        return default_song(title, artist), [
            f"could not estimate features ({describe_error(exc)}); please set them manually"]


# ---------- Full pipeline ----------

def new_result(request: str) -> Dict:
    return {"request": request, "prefs": dict(DEFAULT_PREFS), "profile_source": "defaults",
            "retrieved": [], "summary": "", "picks": [], "used_ai": False, "warnings": []}


def connect(client: Optional[LLMClient], result: Dict, on_status) -> Optional[LLMClient]:
    """Return a usable client (creating Gemini if needed), or None with a warning."""
    if client is None:
        try:
            client = GeminiClient()
        except Exception as exc:  # missing key or SDK
            result["warnings"].append(f"AI unavailable ({exc})")
            log.warning("AI unavailable: %s", exc)
            return None
    if hasattr(client, "on_status"):
        client.on_status = on_status
    return client


def clean_request(request: str) -> str:
    request = (request or "").strip()
    if len(request) > MAX_REQUEST_CHARS:
        log.warning("request truncated from %d chars", len(request))
        request = request[:MAX_REQUEST_CHARS]
    return request


def try_parse(request: str, songs: List[Dict], client, result: Dict, status) -> Optional[Dict]:
    """Steps 1-2: parse + validate. Returns prefs, or None (with a warning) on failure."""
    status("Understanding your request...")
    try:
        prefs, warns = parse_request(request, songs, client)
        result["warnings"] += warns
        return prefs
    except Exception as exc:
        result["warnings"].append(f"could not understand request ({describe_error(exc)})")
        log.error("parse failed: %s", exc)
        return None


def retrieve_and_answer(result: Dict, ask: str, pool: List[Dict], client, k: int,
                        status, exclude_ids=frozenset()) -> Dict:
    """Steps 3-5: retrieve with the weighted scorer, generate a grounded answer, verify."""
    log.info("profile (%s): %s", result["profile_source"], result["prefs"])
    status("Searching the catalog...")
    candidates = [s for s in pool if s["id"] not in exclude_ids]
    retrieved = recommend_songs(result["prefs"], candidates, k)
    result["retrieved"] = retrieved
    log.info("retrieved: %s", [s["title"] for s, _, _ in retrieved])

    if client is not None and retrieved:
        status("Writing recommendations...")
        try:
            answer = extract_json(client.generate(build_answer_prompt(ask, retrieved)))
            picks, warns = verify_picks(answer, retrieved)
            result["warnings"] += warns
            if picks:
                result.update(summary=str(answer.get("summary", "")).strip(),
                              picks=picks, used_ai=True)
            else:
                result["warnings"].append("LLM returned no valid picks; using ranked list")
        except Exception as exc:
            result["warnings"].append(
                f"could not generate answer ({describe_error(exc)}); using ranked list")
            log.error("generation failed: %s", exc)

    if not result["picks"]:
        result["summary"] = "Here are the closest matches from the catalog."
        result["picks"] = fallback_picks(retrieved)

    for w in result["warnings"]:
        log.warning(w)
    log.info("final picks (used_ai=%s): %s", result["used_ai"],
             [p["song"]["title"] for p in result["picks"]])
    return result


def recommend_from_text(request: str, songs: List[Dict],
                        client: Optional[LLMClient] = None, k: int = 5,
                        on_status: Optional[Callable[[str], None]] = None) -> Dict:
    """
    Run the full RAG pipeline for a text request. Never raises for AI problems;
    instead returns a result with used_ai=False and warnings explaining what went wrong.
    """
    request = clean_request(request)
    if not request:
        raise ValueError("request must not be empty")
    status = on_status or (lambda msg: None)
    log.info("request: %r", request)
    result = new_result(request)

    client = connect(client, result, on_status)
    if client is not None:
        prefs = try_parse(request, songs, client, result, status)
        if prefs is not None:
            result.update(prefs=prefs, profile_source="request")
    if result["profile_source"] == "defaults":
        result["warnings"].append("using default preferences")
    return retrieve_and_answer(result, request, songs, client, k, status)


def blend_prefs(base: Dict, tweak: Dict) -> Dict:
    """Mix a playlist profile with a request: numbers 50/50, request genre/mood win if set."""
    blended = {key: round((base[key] + tweak[key]) / 2, 3) for key in NUMERIC_KEYS}
    for key in ("favorite_genre", "favorite_mood"):
        blended[key] = tweak[key] or base[key]
    return blended


def recommend_for_playlist(name: str, profile: Dict, playlist_ids, pool: List[Dict],
                           client: Optional[LLMClient] = None, request: str = "", k: int = 5,
                           on_status: Optional[Callable[[str], None]] = None) -> Dict:
    """
    Recommend songs that fit a playlist's taste profile (optionally tweaked by a text
    request), excluding songs already in the playlist.
    """
    request = clean_request(request)
    status = on_status or (lambda msg: None)
    log.info("playlist recommendation: %r (tweak: %r)", name, request)
    result = new_result(request)
    result.update(prefs=dict(profile), profile_source=f"playlist '{name}'")

    client = connect(client, result, on_status)
    if client is not None and request:
        tweak = try_parse(request, pool, client, result, status)
        if tweak is not None:
            result.update(prefs=blend_prefs(profile, tweak),
                          profile_source=f"playlist '{name}' + your request")
    elif request:
        result["warnings"].append("request ignored because the AI is unavailable")

    ask = f"more songs that fit my playlist '{name}'" + (f", {request}" if request else "")
    return retrieve_and_answer(result, ask, pool, client, k, status, frozenset(playlist_ids))
