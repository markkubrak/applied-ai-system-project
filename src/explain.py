"""
Plain-English explanations of profiles and scores, for people who don't want raw numbers.

Mirrors the scoring rules in src/recommender.py: genre and mood are exact matches,
the numeric features score by closeness to the user's target, popularity by discovery.
"""

from typing import Dict, List

MAX_SCORE = 6.75  # sum of all weights in src/recommender.py

# feature -> (prefs key, friendly name, word for low values, word for high values)
FEATURES = {
    "energy": ("target_energy", "Energy", "calm", "intense"),
    "valence": ("target_valence", "Mood tone", "dark / sad", "bright / happy"),
    "acousticness": ("target_acousticness", "Sound", "electronic", "acoustic"),
    "instrumentalness": ("target_instrumentalness", "Vocals", "mostly vocals", "instrumental"),
}
LEVELS = [(0.2, "very low"), (0.4, "low"), (0.6, "medium"), (0.8, "high"), (1.01, "very high")]


def match_percent(score: float) -> int:
    """Score as a 0-100% match against the highest possible score."""
    return max(0, min(100, round(score / MAX_SCORE * 100)))


def level(value: float) -> str:
    """Bucket a 0-1 value into a word: very low / low / medium / high / very high."""
    for limit, word in LEVELS:
        if value < limit:
            return word
    return "very high"


def describe(value: float, low_word: str, high_word: str) -> str:
    """Describe a 0-1 value with feature words, e.g. 0.9 energy -> 'very intense'."""
    if value < 0.2:
        return f"very {low_word}"
    if value < 0.4:
        return low_word
    if value < 0.6:
        return "in between"
    if value < 0.8:
        return high_word
    return f"very {high_word}"


def profile_words(prefs: Dict) -> List[str]:
    """A profile as friendly labels, e.g. ['Genre: rock', 'Energy: very intense', ...]."""
    words = [f"Genre: {prefs.get('favorite_genre') or 'any'}",
             f"Mood: {prefs.get('favorite_mood') or 'any'}"]
    for _, (key, name, low, high) in FEATURES.items():
        if key in prefs:
            words.append(f"{name}: {describe(prefs[key], low, high)}")
    discovery = prefs.get("discovery", 0.5)
    words.append("Prefers: hidden gems" if discovery >= 0.65 else
                 "Prefers: popular hits" if discovery <= 0.35 else "Prefers: a mix of hits and gems")
    return words


def friendly_reasons(song: Dict, prefs: Dict) -> List[str]:
    """Why a song matched, in plain English (no numbers)."""
    reasons = []
    if song.get("genre") and song["genre"] == prefs.get("favorite_genre"):
        reasons.append(f"✓ Same genre ({song['genre']})")
    if song.get("mood") and song["mood"] == prefs.get("favorite_mood"):
        reasons.append(f"✓ Same mood ({song['mood']})")
    for field, (key, name, low, high) in FEATURES.items():
        if key not in prefs or field not in song:
            continue
        diff = float(song[field]) - float(prefs[key])
        if abs(diff) < 0.1:
            reasons.append(f"✓ {name}: very close to what you want")
        elif abs(diff) < 0.25:
            direction = high if diff > 0 else low
            reasons.append(f"{name}: close, a bit more {direction}")
        else:
            direction = high if diff > 0 else low
            reasons.append(f"✗ {name}: much more {direction} than you want")
    if "popularity" in song:
        popular = song["popularity"] >= 60
        wants_gems = prefs.get("discovery", 0.5) >= 0.5
        tag = "popular hit" if popular else "hidden gem (less popular)"
        mark = "✓ " if popular != wants_gems else ""
        reasons.append(f"{mark}It's a {tag}")
    return reasons
