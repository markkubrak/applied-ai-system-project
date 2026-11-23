"""Tests for the plain-English explanation helpers."""

from src.explain import describe, friendly_reasons, level, match_percent, profile_words

PREFS = {"favorite_genre": "rock", "favorite_mood": "intense", "target_energy": 0.9,
         "target_acousticness": 0.1, "target_valence": 0.4,
         "target_instrumentalness": 0.2, "discovery": 0.8}


def test_match_percent_is_bounded():
    assert match_percent(6.75) == 100
    assert match_percent(3.375) == 50
    assert match_percent(-1) == 0 and match_percent(99) == 100


def test_level_and_describe_buckets():
    assert level(0.05) == "very low" and level(0.5) == "medium" and level(1.0) == "very high"
    assert describe(0.95, "calm", "intense") == "very intense"
    assert describe(0.3, "calm", "intense") == "calm"


def test_profile_words_are_readable():
    words = profile_words(PREFS)
    assert "Genre: rock" in words
    assert "Energy: very intense" in words
    assert "Prefers: hidden gems" in words


def test_friendly_reasons_for_close_match():
    song = {"genre": "rock", "mood": "intense", "energy": 0.92, "acousticness": 0.1,
            "valence": 0.45, "instrumentalness": 0.15, "popularity": 30}
    reasons = friendly_reasons(song, PREFS)
    assert "✓ Same genre (rock)" in reasons
    assert "✓ Energy: very close to what you want" in reasons
    assert "✓ It's a hidden gem (less popular)" in reasons


def test_friendly_reasons_flags_far_values():
    song = {"genre": "lofi", "mood": "chill", "energy": 0.3, "acousticness": 0.1,
            "valence": 0.4, "instrumentalness": 0.2, "popularity": 90}
    reasons = friendly_reasons(song, PREFS)
    assert "✗ Energy: much more calm than you want" in reasons
    assert not any("Same genre" in r for r in reasons)
    assert "It's a popular hit" in reasons
