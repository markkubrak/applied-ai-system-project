"""
User music library: songs the user adds plus named playlists, saved to JSON.

A playlist is turned into a taste profile (the same prefs dict that score_song uses)
by averaging its songs' features, so the weighted scoring engine works unchanged.
"""

import json
import logging
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List

log = logging.getLogger("music_ai")  # shared logger, configured in ai_recommender

LIBRARY_PATH = Path("data/user_library.json")
FIRST_USER_ID = 1001  # catalog ids are small; user ids never collide with them
PROFILE_FEATURES = {
    "target_energy": "energy",
    "target_acousticness": "acousticness",
    "target_valence": "valence",
    "target_instrumentalness": "instrumentalness",
}
ALL_PLAYLISTS = "All playlists"


def empty_library() -> Dict:
    return {"songs": [], "playlists": {}}


def load_library(path: Path = LIBRARY_PATH) -> Dict:
    """Load the library; a missing file gives an empty one, a corrupt file is backed up."""
    path = Path(path)
    if not path.exists():
        return empty_library()
    try:
        lib = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(lib.get("songs"), list) or not isinstance(lib.get("playlists"), dict):
            raise ValueError("unexpected structure")
        return lib
    except (ValueError, OSError) as exc:
        backup = path.with_suffix(".json.bak")
        shutil.copy(path, backup)
        log.error("library file corrupt (%s); backed up to %s and started fresh", exc, backup)
        return empty_library()


def save_library(lib: Dict, path: Path = LIBRARY_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lib, indent=2), encoding="utf-8")


def find_song(lib: Dict, title: str, artist: str):
    """Return the user song with this title and artist (case-insensitive), or None."""
    key = (title.strip().lower(), artist.strip().lower())
    for song in lib["songs"]:
        if (song["title"].lower(), song["artist"].lower()) == key:
            return song
    return None


def add_song(lib: Dict, song: Dict) -> Dict:
    """Add a song (or return the existing one with the same title + artist)."""
    title, artist = str(song.get("title", "")).strip(), str(song.get("artist", "")).strip()
    if not title or not artist:
        raise ValueError("a song needs both a title and an artist")
    existing = find_song(lib, title, artist)
    if existing:
        return existing
    next_id = max([FIRST_USER_ID - 1] + [s["id"] for s in lib["songs"]]) + 1
    new = {**song, "id": next_id, "title": title, "artist": artist, "source": "user"}
    lib["songs"].append(new)
    log.info("added user song %s: %s by %s", next_id, title, artist)
    return new


def create_playlist(lib: Dict, name: str) -> None:
    name = name.strip()
    if not name:
        raise ValueError("playlist name must not be empty")
    if name == ALL_PLAYLISTS:
        raise ValueError(f"'{ALL_PLAYLISTS}' is a reserved name")
    lib["playlists"].setdefault(name, [])


def delete_playlist(lib: Dict, name: str) -> None:
    lib["playlists"].pop(name, None)


def add_to_playlist(lib: Dict, name: str, song_id: int) -> None:
    if name not in lib["playlists"]:
        raise KeyError(f"no playlist named '{name}'")
    if song_id not in lib["playlists"][name]:
        lib["playlists"][name].append(song_id)


def remove_from_playlist(lib: Dict, name: str, song_id: int) -> None:
    if song_id in lib["playlists"].get(name, []):
        lib["playlists"][name].remove(song_id)


def all_songs(catalog: List[Dict], lib: Dict) -> List[Dict]:
    """The recommendation pool: catalog songs plus the user's own songs."""
    return catalog + lib["songs"]


def playlist_songs(lib: Dict, name: str, pool: List[Dict]) -> List[Dict]:
    """Songs in a playlist (or in every playlist, for ALL_PLAYLISTS), in order, no repeats."""
    if name == ALL_PLAYLISTS:
        ids = [i for ids in lib["playlists"].values() for i in ids]
    elif name in lib["playlists"]:
        ids = lib["playlists"][name]
    else:
        raise KeyError(f"no playlist named '{name}'")
    by_id = {s["id"]: s for s in pool}
    return [by_id[i] for i in dict.fromkeys(ids) if i in by_id]


def most_common(values: List[str]) -> str:
    """The single most common value, or "" when empty or tied (avoids a false match bonus)."""
    counts = Counter(v for v in values if v).most_common(2)
    if not counts or (len(counts) == 2 and counts[0][1] == counts[1][1]):
        return ""
    return counts[0][0]


def playlist_profile(songs: List[Dict]) -> Dict:
    """Build a taste profile (prefs dict for score_song) from a list of songs."""
    if not songs:
        raise ValueError("playlist is empty; add songs before asking for recommendations")
    n = len(songs)
    prefs = {key: round(sum(float(s.get(field, 0.5)) for s in songs) / n, 3)
             for key, field in PROFILE_FEATURES.items()}
    prefs["favorite_genre"] = most_common([s.get("genre", "") for s in songs])
    prefs["favorite_mood"] = most_common([s.get("mood", "") for s in songs])
    mean_pop = sum(float(s.get("popularity", 50)) for s in songs) / n
    prefs["discovery"] = round(1 - mean_pop / 100, 3)  # obscure playlist -> favor hidden gems
    return prefs
