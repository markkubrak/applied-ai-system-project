"""
Command line runner for the Music Recommender Simulation.

Run from the project root with:
    python -m src.main                      interactive AI recommender
    python -m src.main --ask "chill study music, no vocals"
    python -m src.main --profiles           original preset-profile demo (no AI)
    python -m src.main --playlist Gym       recommend for a saved playlist ("all" = every playlist)
    streamlit run app.py                    web app: playlists, add songs, mood mode
"""

import argparse

from dotenv import load_dotenv

from src.ai_recommender import recommend_for_playlist, recommend_from_text
from src.library import (ALL_PLAYLISTS, all_songs, load_library, playlist_profile,
                         playlist_songs)
from src.recommender import load_songs, recommend_songs

# Highest possible score: sum of all weights in src/recommender.py (see README table)
MAX_SCORE = 6.75

# High-Energy Pop (default): upbeat pop listener
POP_HAPPY = {
    "favorite_genre": "pop",
    "favorite_mood": "happy",
    "target_energy": 0.80,            # 0.0-1.0, scored by closeness
    "target_acousticness": 0.20,      # prefers produced/electronic texture
    "target_valence": 0.80,           # bright and positive
    "target_instrumentalness": 0.05,  # wants vocals
    "discovery": 0.5,                 # 0 = favor hits, 1 = favor hidden gems
}

# Chill Lofi: study listener
LOFI_CHILL = {
    "favorite_genre": "lofi",
    "favorite_mood": "chill",
    "target_energy": 0.40,
    "target_acousticness": 0.80,
    "target_valence": 0.60,
    "target_instrumentalness": 0.80,
    "discovery": 0.5,
}

# Deep Intense Rock: loud, dark, guitar-driven
ROCK_INTENSE = {
    "favorite_genre": "rock",
    "favorite_mood": "intense",
    "target_energy": 0.92,
    "target_acousticness": 0.10,
    "target_valence": 0.35,
    "target_instrumentalness": 0.15,
    "discovery": 0.5,
}

# --- Adversarial / edge-case profiles ---

# Conflicting: very high energy but a sad mood (sad songs in the catalog are low energy)
SAD_BUT_HYPED = {
    "favorite_genre": "blues",
    "favorite_mood": "sad",
    "target_energy": 0.90,
    "target_acousticness": 0.20,
    "target_valence": 0.20,
    "target_instrumentalness": 0.20,
    "discovery": 0.5,
}

# Genre and mood that do not exist in the catalog: only numeric features can score
GENRE_GHOST = {
    "favorite_genre": "k-pop",
    "favorite_mood": "dreamy",
    "target_energy": 0.65,
    "target_acousticness": 0.30,
    "target_valence": 0.70,
    "target_instrumentalness": 0.10,
    "discovery": 0.5,
}

# Contradictory texture: wants EDM but fully acoustic and instrumental
ACOUSTIC_RAVER = {
    "favorite_genre": "edm",
    "favorite_mood": "euphoric",
    "target_energy": 0.95,
    "target_acousticness": 0.95,
    "target_valence": 0.75,
    "target_instrumentalness": 0.90,
    "discovery": 0.5,
}

# Middle-of-the-road taste, maximum discovery: does popularity decide everything?
HIDDEN_GEM_HUNTER = {
    "favorite_genre": "",
    "favorite_mood": "",
    "target_energy": 0.50,
    "target_acousticness": 0.50,
    "target_valence": 0.50,
    "target_instrumentalness": 0.50,
    "discovery": 1.0,
}

PROFILES = {
    "High-Energy Pop (default)": POP_HAPPY,
    "Chill Lofi": LOFI_CHILL,
    "Deep Intense Rock": ROCK_INTENSE,
    "Edge case: Sad but Hyped": SAD_BUT_HYPED,
    "Edge case: Genre Ghost": GENRE_GHOST,
    "Edge case: Acoustic Raver": ACOUSTIC_RAVER,
    "Edge case: Hidden Gem Hunter": HIDDEN_GEM_HUNTER,
}


def print_recommendations(name: str, user_prefs: dict, songs: list, k: int = 5) -> None:
    """Print a ranked, readable list of recommendations with score breakdowns."""
    print("=" * 64)
    print(f"Profile: {name}")
    print("=" * 64)
    for rank, (song, score, explanation) in enumerate(recommend_songs(user_prefs, songs, k), 1):
        print(f"{rank}. {song['title']} by {song['artist']}  [{song['genre']}, {song['mood']}]")
        print(f"   Score: {score:.2f}")
        for reason in explanation.split("; "):
            print(f"     - {reason}")
    print()


def show_status(message: str) -> None:
    """Print a progress line right away so slow API calls don't look frozen."""
    print(f"  ... {message}", flush=True)


def print_ai_result(result: dict) -> None:
    """Print the output of the AI (RAG) pipeline."""
    mode = "AI (Gemini + retrieval)" if result["used_ai"] else "Fallback (ranked list, no AI)"
    print("=" * 64)
    print(f"Request: {result['request'] or '(none)'}")
    print(f"Mode: {mode}")
    print("=" * 64)

    # 1. The user profile the AI built (same shape as the preset profiles)
    p = result["prefs"]
    source = result["profile_source"]
    print(f"Profile from {source}:" if source != "defaults"
          else "Profile (defaults; the AI could not read your request):")
    print(f"  genre: {p['favorite_genre'] or '(any)'} | mood: {p['favorite_mood'] or '(any)'}")
    print(f"  energy {p['target_energy']:.2f} | acousticness {p['target_acousticness']:.2f} | "
          f"valence {p['target_valence']:.2f} | instrumentalness "
          f"{p['target_instrumentalness']:.2f} | discovery {p['discovery']:.2f}")

    # 2. Final picks (AI-written when available), with each song's weighted score
    scores = {song["id"]: score for song, score, _ in result["retrieved"]}
    print(f"\n{'AI picks' if result['used_ai'] else 'Top picks'}: {result['summary']}")
    for rank, pick in enumerate(result["picks"], 1):
        song = pick["song"]
        print(f"{rank}. {song['title']} by {song['artist']}  [{song['genre']}, {song['mood']}]"
              f"  score {scores[song['id']]:.2f}")
        if result["used_ai"]:
            print(f"   {pick['reason']}")

    # 3. What the weighted scoring engine retrieved, with full breakdowns
    picked = {pick["song"]["id"] for pick in result["picks"]}
    print(f"\nRetrieved by weighted scoring (top {len(result['retrieved'])}, "
          f"max {MAX_SCORE:.2f}; * = picked):")
    for rank, (song, score, explanation) in enumerate(result["retrieved"], 1):
        mark = "*" if song["id"] in picked else " "
        print(f" {mark}{rank}. {song['title']} by {song['artist']}  score {score:.2f}")
        for reason in explanation.split("; "):
            print(f"       - {reason}")

    if result["warnings"]:
        print("\nNotes:")
        for w in result["warnings"]:
            print(f"  - {w}")
    print()


def run_playlist(name: str, tweak: str, catalog: list) -> None:
    """Recommend for a saved playlist (created in the web app)."""
    lib = load_library()
    name = ALL_PLAYLISTS if name.lower() == "all" else name
    pool = all_songs(catalog, lib)
    try:
        songs = playlist_songs(lib, name, pool)
        profile = playlist_profile(songs)
    except (KeyError, ValueError) as exc:
        names = ", ".join(lib["playlists"]) or "none yet (create one with: streamlit run app.py)"
        print(f"Error: {exc.args[0]}. Saved playlists: {names}")
        return
    print(f"Playlist '{name}': {len(songs)} songs")
    print_ai_result(recommend_for_playlist(name, profile, [s["id"] for s in songs], pool,
                                           request=tweak, on_status=show_status))


def main() -> None:
    parser = argparse.ArgumentParser(description="Music Recommender Simulation")
    parser.add_argument("--ask", metavar="REQUEST",
                        help='natural-language request, e.g. --ask "calm music for studying"')
    parser.add_argument("--profiles", action="store_true",
                        help="run the original preset-profile demo (no AI)")
    parser.add_argument("--playlist", metavar="NAME",
                        help='recommend for a saved playlist; "all" combines every playlist')
    parser.add_argument("--tweak", metavar="REQUEST", default="",
                        help='with --playlist: adjust the taste, e.g. --tweak "more acoustic"')
    args = parser.parse_args()

    load_dotenv()  # reads GEMINI_API_KEY from .env if present
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}\n")

    if args.profiles:
        for name, prefs in PROFILES.items():
            print_recommendations(name, prefs, songs)
    elif args.playlist:
        run_playlist(args.playlist, args.tweak, songs)
    elif args.ask:
        print_ai_result(recommend_from_text(args.ask, songs, on_status=show_status))
    else:
        print("Describe what you want to hear (blank line to quit).")
        while True:
            try:
                request = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not request:
                break
            print_ai_result(recommend_from_text(request, songs, on_status=show_status))


if __name__ == "__main__":
    main()
