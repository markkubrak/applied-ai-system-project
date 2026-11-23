"""Tests for playlists, user songs, feature estimation, and playlist recommendations."""

import pytest

from src.ai_recommender import (blend_prefs, estimate_song_features,
                                recommend_for_playlist, validate_song)
from src.library import (ALL_PLAYLISTS, add_song, add_to_playlist, all_songs,
                         create_playlist, empty_library, load_library, most_common,
                         playlist_profile, playlist_songs, save_library)
from src.recommender import load_songs
from tests.test_ai_recommender import FakeLLM

CATALOG = load_songs("data/songs.csv")


def song(title, genre="rock", mood="intense", energy=0.9, popularity=50, **extra):
    return {"title": title, "artist": "Tester", "genre": genre, "mood": mood,
            "energy": energy, "acousticness": 0.1, "valence": 0.4,
            "instrumentalness": 0.2, "popularity": popularity, "tempo_bpm": 140, **extra}


def test_playlist_profile_averages_features():
    profile = playlist_profile([song("A", energy=0.8, popularity=20),
                                song("B", energy=0.6, popularity=40)])
    assert profile["target_energy"] == 0.7
    assert profile["favorite_genre"] == "rock"
    assert profile["favorite_mood"] == "intense"
    assert profile["discovery"] == 0.7  # mean popularity 30 -> favors hidden gems


def test_tied_genre_gives_no_genre_preference():
    assert most_common(["rock", "pop"]) == ""
    assert most_common(["rock", "pop", "rock"]) == "rock"


def test_empty_playlist_is_rejected():
    with pytest.raises(ValueError):
        playlist_profile([])


def test_add_song_assigns_user_id_and_skips_duplicates():
    lib = empty_library()
    first = add_song(lib, song("My Track"))
    again = add_song(lib, song("my track"))  # same title, different case
    assert first["id"] >= 1001 and first["source"] == "user"
    assert again is first and len(lib["songs"]) == 1


def test_all_playlists_combines_without_repeats():
    lib = empty_library()
    s1, s2 = add_song(lib, song("One")), add_song(lib, song("Two"))
    for name, ids in (("Gym", [s1["id"], s2["id"]]), ("Run", [s2["id"]])):
        create_playlist(lib, name)
        for i in ids:
            add_to_playlist(lib, name, i)
    pool = all_songs(CATALOG, lib)
    assert [s["title"] for s in playlist_songs(lib, ALL_PLAYLISTS, pool)] == ["One", "Two"]


def test_library_save_load_and_corrupt_file_recovery(tmp_path):
    path = tmp_path / "lib.json"
    lib = empty_library()
    create_playlist(lib, "Chill")
    save_library(lib, path)
    assert load_library(path)["playlists"] == {"Chill": []}

    path.write_text("{not json", encoding="utf-8")
    assert load_library(path) == empty_library()
    assert (tmp_path / "lib.json.bak").exists()


def test_playlist_recommendations_exclude_playlist_songs():
    playlist = [s for s in CATALOG if s["genre"] == "lofi"]
    profile = playlist_profile(playlist)
    llm = FakeLLM({"summary": "More lofi.", "picks": [{"title": "Spacewalk Thoughts", "reason": "calm"}]})
    result = recommend_for_playlist("Study", profile, [s["id"] for s in playlist], CATALOG,
                                    client=llm)

    retrieved_ids = {s["id"] for s, _, _ in result["retrieved"]}
    assert not retrieved_ids & {s["id"] for s in playlist}
    assert result["used_ai"] and result["profile_source"] == "playlist 'Study'"


def test_request_tweak_blends_with_playlist_profile():
    base = {"favorite_genre": "rock", "favorite_mood": "intense", "target_energy": 0.9,
            "target_acousticness": 0.1, "target_valence": 0.4,
            "target_instrumentalness": 0.2, "discovery": 0.5}
    tweak = {**base, "favorite_genre": "", "favorite_mood": "chill", "target_acousticness": 0.9}
    blended = blend_prefs(base, tweak)
    assert blended["target_acousticness"] == 0.5
    assert blended["favorite_genre"] == "rock"   # tweak left genre blank
    assert blended["favorite_mood"] == "chill"   # tweak set a mood


def test_validate_song_clamps_and_flags_unknown_songs():
    raw = {"known": False, "genre": "Hyperpop", "mood": "", "energy": 1.4,
           "tempo_bpm": 400, "popularity": "lots"}
    estimated, warnings = validate_song(raw, "X", "Y")
    assert estimated["genre"] == "hyperpop"
    assert estimated["energy"] == 1.0 and estimated["tempo_bpm"] == 220
    assert estimated["popularity"] == 50
    assert any("doesn't recognize" in w for w in warnings)


def test_estimate_falls_back_to_manual_on_api_error():
    estimated, warnings = estimate_song_features("X", "Y", CATALOG,
                                                 client=FakeLLM(RuntimeError("503 UNAVAILABLE")))
    assert estimated["estimated"] is False and estimated["energy"] == 0.5
    assert any("set them manually" in w for w in warnings)
