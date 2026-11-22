"""Tests for the RAG pipeline. A fake LLM stands in for Gemini, so no API key is needed."""

import json

from src.ai_recommender import recommend_from_text, validate_prefs, extract_json
from src.recommender import load_songs

SONGS = load_songs("data/songs.csv")
GENRES = sorted({s["genre"] for s in SONGS})
MOODS = sorted({s["mood"] for s in SONGS})


class FakeLLM:
    """Returns scripted responses in order and records every prompt it receives."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, str) else json.dumps(item)


LOFI_PREFS = {"favorite_genre": "lofi", "favorite_mood": "chill", "target_energy": 0.35,
              "target_acousticness": 0.8, "target_valence": 0.6,
              "target_instrumentalness": 0.9, "discovery": 0.5}


def test_ai_answer_is_grounded_in_retrieved_songs():
    llm = FakeLLM(LOFI_PREFS, {"summary": "Calm picks.", "picks": [
        {"title": "Library Rain", "reason": "Quiet and instrumental."}]})
    result = recommend_from_text("calm study music", SONGS, client=llm)

    assert result["used_ai"] is True
    assert result["picks"][0]["song"]["title"] == "Library Rain"
    # Retrieval used the parsed prefs, and the answer prompt contained the retrieved songs
    assert result["retrieved"][0][0]["genre"] == "lofi"
    for song, _, _ in result["retrieved"]:
        assert song["title"] in llm.prompts[1]


def test_hallucinated_song_is_dropped():
    llm = FakeLLM(LOFI_PREFS, {"summary": "x", "picks": [
        {"title": "Totally Made Up Song", "reason": "?"},
        {"title": "Focus Flow", "reason": "Good for focus."}]})
    result = recommend_from_text("study music", SONGS, client=llm)

    titles = [p["song"]["title"] for p in result["picks"]]
    assert titles == ["Focus Flow"]
    assert any("Totally Made Up Song" in w for w in result["warnings"])


def test_only_hallucinations_falls_back_to_ranked_list():
    llm = FakeLLM(LOFI_PREFS, {"summary": "x", "picks": [{"title": "Fake", "reason": "?"}]})
    result = recommend_from_text("study music", SONGS, client=llm)

    assert result["used_ai"] is False
    assert len(result["picks"]) == 3
    assert result["picks"][0]["song"] == result["retrieved"][0][0]


def test_llm_errors_fall_back_safely():
    llm = FakeLLM(RuntimeError("network down"), RuntimeError("network down"))
    result = recommend_from_text("anything", SONGS, client=llm)

    assert result["used_ai"] is False
    assert result["picks"]
    assert any("could not understand" in w for w in result["warnings"])


def test_validate_prefs_clamps_and_rejects_unknown_values():
    raw = {"favorite_genre": "k-pop", "favorite_mood": "CHILL", "target_energy": 1.7,
           "target_valence": "loud", "discovery": -0.2}
    prefs, warnings = validate_prefs(raw, GENRES, MOODS)

    assert prefs["favorite_genre"] == ""
    assert prefs["favorite_mood"] == "chill"
    assert prefs["target_energy"] == 1.0
    assert prefs["target_valence"] == 0.5
    assert prefs["discovery"] == 0.0
    assert len(warnings) == 4


def test_extract_json_handles_code_fences():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_empty_request_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        recommend_from_text("   ", SONGS, client=FakeLLM())


def test_status_messages_report_each_step():
    llm = FakeLLM(LOFI_PREFS, {"summary": "x", "picks": [{"title": "Focus Flow", "reason": "ok"}]})
    messages = []
    recommend_from_text("study music", SONGS, client=llm, on_status=messages.append)

    assert messages == ["Understanding your request...", "Searching the catalog...",
                        "Writing recommendations..."]
