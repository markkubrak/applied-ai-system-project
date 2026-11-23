"""
Streamlit web app for the AI music recommender.

Run from the project root with:  streamlit run app.py

Tabs:
  Discover      - describe what you want; Gemini + weighted retrieval recommend songs
  My Playlists  - build playlists; get recommendations that fit a playlist's taste
  Add a Song    - add your own song; Gemini estimates its features, you can edit them
Sidebar "Look" setting: Off | Mood | Mood (animated) - background reflects the mood.
"""

import streamlit as st
from dotenv import load_dotenv

from src.ai_recommender import (default_song, estimate_song_features,
                                recommend_for_playlist, recommend_from_text)
from src.library import (ALL_PLAYLISTS, add_song, add_to_playlist, all_songs,
                         create_playlist, delete_playlist, load_library,
                         playlist_profile, playlist_songs, remove_from_playlist,
                         save_library)
from src.explain import MAX_SCORE, describe, friendly_reasons, match_percent, profile_words
from src.recommender import load_songs

LOOKS = ["Off", "Mood", "Mood (animated)"]

load_dotenv()
st.set_page_config(page_title="Mood Music Recommender", page_icon="🎵", layout="wide")


@st.cache_data
def get_catalog():
    return load_songs("data/songs.csv")


# ---------------------------------------------------------------- mood styling

def mood_colors(prefs: dict) -> tuple:
    """Map a profile to two background colors: energy picks the hue, valence the brightness."""
    valence = prefs.get("target_valence", 0.5)
    energy = prefs.get("target_energy", 0.5)
    hue = 230 - (energy ** 1.5) * 215        # calm = blue -> teal -> intense = orange-red
    sat = 45 + energy * 40                   # calm = softer, intense = vivid
    light = 18 + valence * 16                # sad = darker, happy = brighter
    c1 = f"hsl({hue:.0f}, {sat:.0f}%, {light:.0f}%)"
    c2 = f"hsl({(hue + 40) % 360:.0f}, {sat:.0f}%, {light - 8:.0f}%)"
    return c1, c2


def apply_style(look: str, prefs: dict, bpm: float) -> None:
    """Inject CSS: a neutral theme, or a mood gradient (optionally animated)."""
    # Colors are registered CSS properties so the browser can fade between them
    c1, c2 = ("#0f172a", "#0f172a") if look == "Off" else mood_colors(prefs)
    animation = ""
    if look != "Off":
        if look == "Mood (animated)":
            drift = 30 - 22 * prefs.get("target_energy", 0.5)   # seconds per cycle
            beat = 60 / max(40, min(220, bpm))                   # one pulse per beat
            animation = f"""
            .stApp {{ animation: drift {drift:.1f}s ease-in-out infinite; }}
            .stApp::before {{
                content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
                background: radial-gradient(circle at 50% 40%, rgba(255,255,255,0.18), transparent 60%);
                animation: pulse {beat:.3f}s ease-in-out infinite;
            }}
            @keyframes drift {{ 0% {{background-position: 0% 50%;}} 50% {{background-position: 100% 50%;}}
                               100% {{background-position: 0% 50%;}} }}
            @keyframes pulse {{ 0%, 100% {{opacity: 0.25;}} 50% {{opacity: 1;}} }}
            @media (prefers-reduced-motion: reduce) {{
                .stApp, .stApp::before {{ animation: none !important; }}
            }}"""
    st.markdown(f"""<style>
        @property --mood1 {{ syntax: '<color>'; inherits: false; initial-value: #0f172a; }}
        @property --mood2 {{ syntax: '<color>'; inherits: false; initial-value: #0f172a; }}
        .stApp {{
            --mood1: {c1}; --mood2: {c2};
            background: linear-gradient(135deg, var(--mood1), var(--mood2), var(--mood1));
            background-size: 300% 300%;
            transition: --mood1 1.2s ease, --mood2 1.2s ease;
        }}
        .st-key-confirm_delete button {{ background: #dc2626; border-color: #dc2626; color: #ffffff; }}
        .match {{ font-weight: 700; color: #4c1d95; }}
        [data-testid="stHeader"] {{ background: transparent; }}
        /* All content sits on a solid card so text contrast never depends on the background */
        .block-container {{
            background: #ffffff; color: #111827; border-radius: 18px;
            padding: 2rem 2.2rem !important; margin-top: 2.5rem; max-width: 1100px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.35); position: relative; z-index: 1;
        }}
        [data-testid="stSidebar"] {{ background: #ffffff; }}
        .pill {{ display: inline-block; padding: 2px 10px; margin: 2px 4px 2px 0;
                 border-radius: 999px; background: #ede9fe; color: #4c1d95; font-size: 0.85rem; }}
        .score {{ font-weight: 700; color: #4c1d95; }}
        {animation}
    </style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- shared pieces

def flash(message: str) -> None:
    """Queue a toast that survives st.rerun() (shown at the start of the next run)."""
    st.session_state["flash"] = message


def save(lib: dict) -> None:
    save_library(lib)
    st.session_state["lib_version"] = st.session_state.get("lib_version", 0) + 1


# Optional playlist-table columns: label -> (song field, words for low / high values or None)
EXTRA_COLUMNS = {
    "Genre": ("genre", None), "Mood": ("mood", None),
    "Energy": ("energy", ("calm", "intense")),
    "Mood tone": ("valence", ("dark / sad", "bright / happy")),
    "Sound": ("acousticness", ("electronic", "acoustic")),
    "Vocals": ("instrumentalness", ("mostly vocals", "instrumental")),
    "Tempo": ("tempo_bpm", None), "Popularity": ("popularity", None),
    "Year": ("release_year", None), "Added by you": ("source", None),
}


def song_row(song: dict, columns: list) -> dict:
    """Title + artist, plus the chosen columns (in words unless numbers are switched on)."""
    row = {"Title": song["title"], "Artist": song["artist"]}
    for label in columns:
        field, words = EXTRA_COLUMNS[label]
        value = song.get(field, "")
        if label == "Added by you":
            value = "✓" if value == "user" else ""
        elif label == "Tempo":
            value = f"{value} BPM"
        elif words and not show_numbers():
            value = describe(float(value), *words)
        row[label] = value
    return row


def show_numbers() -> bool:
    return st.session_state.get("show_numbers", False)


def profile_pills(prefs: dict) -> str:
    """Profile as pills: plain words, plus the raw values when numbers are switched on."""
    words = "".join(f'<span class="pill">{w}</span>' for w in profile_words(prefs))
    if not show_numbers():
        return words
    return words + "<br>" + number_pills(prefs)


def number_pills(prefs: dict) -> str:
    items = [f"genre: {prefs['favorite_genre'] or 'any'}", f"mood: {prefs['favorite_mood'] or 'any'}",
             f"energy {prefs['target_energy']:.2f}", f"acoustic {prefs['target_acousticness']:.2f}",
             f"valence {prefs['target_valence']:.2f}",
             f"instrumental {prefs['target_instrumentalness']:.2f}",
             f"discovery {prefs['discovery']:.2f}"]
    return "".join(f'<span class="pill">{i}</span>' for i in items)


def run_with_status(label: str, fn):
    """Run a pipeline call inside st.status, streaming its progress messages."""
    with st.status(label, expanded=False) as box:
        result = fn(lambda msg: box.write(msg))
        box.update(label="Done" if result["used_ai"] else "Done (fallback, no AI)",
                   state="complete")
    return result


def remember_mood(result: dict) -> None:
    """The latest recommendation drives the mood background."""
    st.session_state["mood_source"] = "result"
    st.session_state["mood_prefs"] = result["prefs"]
    if result["picks"]:
        st.session_state["mood_bpm"] = float(result["picks"][0]["song"].get("tempo_bpm", 100))


def add_to_playlist_control(lib: dict, song: dict, key: str) -> None:
    names = list(lib["playlists"])
    if not names:
        st.caption("Create a playlist in **My Playlists** to save songs.")
        return
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    target = c1.selectbox("Playlist", names, key=f"pl_{key}", label_visibility="collapsed")
    if c2.button("Add", key=f"add_{key}"):
        if song["id"] in lib["playlists"][target]:
            flash(f"'{song['title']}' is already in {target}")
        else:
            add_to_playlist(lib, target, song["id"])
            save(lib)
            flash(f"Added '{song['title']}' to {target}")
        st.rerun()  # redraw now so the playlist shows the new song immediately


def render_result(result: dict, lib: dict, key: str) -> None:
    """Show profile, picks and the weighted-scoring breakdown for one pipeline result."""
    source = result["profile_source"]
    st.markdown(f"**Profile from {source}**" if source != "defaults"
                else "**Profile (defaults; the AI could not read your request)**")
    st.markdown(profile_pills(result["prefs"]), unsafe_allow_html=True)

    if result["used_ai"]:
        st.success(result["summary"])
    else:
        st.warning("Gemini unavailable, so these are the top weighted matches. " + result["summary"])

    scores = {s["id"]: (score, why) for s, score, why in result["retrieved"]}
    for i, pick in enumerate(result["picks"]):
        song = pick["song"]
        score, why = scores[song["id"]]
        percent = match_percent(score)
        with st.container(border=True):
            raw = f' <span class="score">({score:.2f} / {MAX_SCORE})</span>' if show_numbers() else ""
            st.markdown(f"### {song['title']}  \n{song['artist']} · "
                        f'<span class="pill">{song["genre"]}</span><span class="pill">{song["mood"]}</span>'
                        f' <span class="match">{percent}% match</span>{raw}',
                        unsafe_allow_html=True)
            st.progress(percent / 100)
            if result["used_ai"]:
                st.write(pick["reason"])
            with st.expander("Why this song?"):
                for reason in friendly_reasons(song, result["prefs"]):
                    st.write(reason)
                if show_numbers():
                    st.caption("Points per feature:")
                    for reason in why.split("; "):
                        st.caption(f"- {reason}")
            add_to_playlist_control(lib, song, f"{key}_{i}")

    with st.expander(f"All {len(result['retrieved'])} songs the matcher found"):
        picked = {p["song"]["id"] for p in result["picks"]}
        rows = []
        for s, sc, _ in result["retrieved"]:
            row = {"picked": "★" if s["id"] in picked else "", "title": s["title"],
                   "artist": s["artist"], "match": f"{match_percent(sc)}%"}
            if show_numbers():
                row["score"] = round(sc, 2)
            rows.append(row)
        st.dataframe(rows, hide_index=True, width="stretch")
    for w in result["warnings"]:
        st.caption(f"Note: {w}")


# ---------------------------------------------------------------- tabs

def discover_tab(catalog: list, lib: dict) -> None:
    st.subheader("What do you want to hear?")
    with st.form("discover"):
        request = st.text_input("Describe it in your own words",
                                placeholder="e.g. upbeat songs for a summer road trip")
        go = st.form_submit_button("Recommend", type="primary")
    if go and request.strip():
        pool = all_songs(catalog, lib)
        result = run_with_status("Finding songs...",
                                 lambda cb: recommend_from_text(request, pool, on_status=cb))
        st.session_state["discover_result"] = result
        remember_mood(result)
        st.rerun()
    if "discover_result" in st.session_state:
        render_result(st.session_state["discover_result"], lib, "disc")


@st.dialog("Delete playlist?")
def confirm_delete(lib: dict, name: str) -> None:
    st.write(f"Delete **{name}**? This removes the playlist. "
             "Your saved songs stay in your library.")
    c1, c2 = st.columns(2)
    if c1.button("Cancel", width="stretch"):
        st.session_state.pop("pending_delete", None)
        st.rerun()
    if c2.button("Delete", key="confirm_delete", width="stretch"):
        delete_playlist(lib, name)
        save(lib)
        st.session_state.pop("pending_delete", None)
        st.session_state.pop("active_playlist", None)
        st.session_state.pop("playlist_result", None)
        st.rerun()


def how_matching_works() -> None:
    with st.expander("How matching works"):
        st.markdown(
            "Every song gets a **match %** against your taste profile:\n"
            "- **Genre** counts the most, then **energy** (calm ↔ intense) and **mood**.\n"
            "- **Sound** (electronic ↔ acoustic), **mood tone** (sad ↔ happy) and "
            "**vocals** (sung ↔ instrumental) fine-tune it: the closer, the better.\n"
            "- **Popularity** breaks ties, toward hits or hidden gems depending on your taste.\n\n"
            "The top matches are handed to Gemini, which picks and explains its favorites. "
            "It can't suggest songs the matcher didn't find.")
        if show_numbers():
            st.table([{"feature": f, "max points": p} for f, p in [
                ("genre match", 2.0), ("energy closeness", 1.5), ("mood match", 1.0),
                ("acousticness closeness", 0.75), ("valence closeness", 0.5),
                ("instrumentalness closeness", 0.5), ("popularity vs discovery", 0.5),
                ("total", 6.75)]])


def playlists_tab(catalog: list, lib: dict) -> None:
    pool = all_songs(catalog, lib)
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Playlists")
        with st.form("new_playlist", clear_on_submit=True):
            name = st.text_input("New playlist name", placeholder="e.g. Gym")
            if st.form_submit_button("Create") and name.strip():
                try:
                    create_playlist(lib, name)
                    save(lib)
                    st.session_state["active_playlist"] = name.strip()
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        names = list(lib["playlists"])
        if not names:
            st.info("No playlists yet. Create one above.")
            return
        options = names + ([ALL_PLAYLISTS] if len(names) > 1 else [])
        if st.session_state.get("active_playlist") not in options:
            st.session_state.pop("active_playlist", None)  # deleted or first visit
        active = st.selectbox("Choose a playlist", options, key="active_playlist")
        if active != ALL_PLAYLISTS and st.button("Delete playlist"):
            st.session_state["pending_delete"] = active
        if st.session_state.get("pending_delete") in lib["playlists"]:
            confirm_delete(lib, st.session_state["pending_delete"])

    with right:
        songs = playlist_songs(lib, active, pool)
        if st.session_state.get("mood_playlist") != active:
            st.session_state["mood_playlist"] = active
            st.session_state["mood_source"] = "playlist"
        if st.session_state.get("mood_source") == "playlist" and songs:
            st.session_state["mood_prefs"] = playlist_profile(songs)
            st.session_state["mood_bpm"] = sum(s.get("tempo_bpm", 100) for s in songs) / len(songs)
        st.subheader(f"{active} · {len(songs)} songs")
        if songs:
            extra = st.pills("Show columns", list(EXTRA_COLUMNS), selection_mode="multi",
                             key="playlist_columns")
            st.dataframe([song_row(s, extra or []) for s in songs],
                         hide_index=True, width="stretch")

        if active != ALL_PLAYLISTS:
            in_list = {s["id"] for s in songs}
            choices = {f"{s['title']} — {s['artist']}": s["id"] for s in pool if s["id"] not in in_list}
            c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
            picked = c1.multiselect("Add songs from the catalog or your library", list(choices))
            if c2.button("Add songs", disabled=not picked):
                for label in picked:
                    add_to_playlist(lib, active, choices[label])
                save(lib)
                st.rerun()
            if songs:
                labels = {f"{s['title']} — {s['artist']}": s["id"] for s in songs}
                c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
                gone = c1.selectbox("Remove a song", list(labels), index=None, placeholder="Choose a song")
                if c2.button("Remove", disabled=gone is None):
                    remove_from_playlist(lib, active, labels[gone])
                    save(lib)
                    st.rerun()

        if not songs:
            st.info("Add songs to this playlist to get recommendations for it.")
            return
        profile = playlist_profile(songs)
        st.markdown("**Taste profile learned from this playlist**")
        st.markdown(profile_pills(profile), unsafe_allow_html=True)
        tweak = st.text_input("Optional tweak", placeholder="e.g. but more acoustic",
                              key=f"tweak_{active}")
        if st.button("Recommend for this playlist", type="primary"):
            ids = [s["id"] for s in songs]
            result = run_with_status("Finding songs that fit...", lambda cb: recommend_for_playlist(
                active, profile, ids, pool, request=tweak, on_status=cb))
            st.session_state["playlist_result"] = result
            remember_mood(result)
            st.rerun()
        if "playlist_result" in st.session_state:
            render_result(st.session_state["playlist_result"], lib, "plr")


def add_song_tab(catalog: list, lib: dict) -> None:
    st.subheader("Add your own song")
    st.caption("Gemini estimates the song's features. Check them and adjust anything that looks off.")
    with st.form("estimate"):
        c1, c2 = st.columns(2)
        title = c1.text_input("Title")
        artist = c2.text_input("Artist")
        c1, c2 = st.columns(2)
        estimate = c1.form_submit_button("Estimate with AI", type="primary")
        manual = c2.form_submit_button("Enter manually")
    if (estimate or manual) and not (title.strip() and artist.strip()):
        st.error("Please enter both a title and an artist.")
    elif estimate:
        with st.spinner("Asking Gemini about this song..."):
            draft, warnings = estimate_song_features(title, artist, all_songs(catalog, lib))
        st.session_state["draft"], st.session_state["draft_warnings"] = draft, warnings
    elif manual:
        st.session_state["draft"] = default_song(title.strip(), artist.strip())
        st.session_state["draft_warnings"] = []

    draft = st.session_state.get("draft")
    if not draft:
        return
    for w in st.session_state.get("draft_warnings", []):
        st.warning(w)
    with st.form("save_song"):
        st.markdown(f"**{draft['title']}** by {draft['artist']}"
                    + (" · *AI-estimated*" if draft.get("estimated") else ""))
        c1, c2 = st.columns(2)
        genre = c1.text_input("Genre", draft["genre"])
        mood = c2.text_input("Mood", draft["mood"])
        c1, c2 = st.columns(2)
        values = {}
        for i, field in enumerate(["energy", "valence", "danceability", "acousticness",
                                   "instrumentalness"]):
            values[field] = (c1 if i % 2 == 0 else c2).slider(field.capitalize(), 0.0, 1.0,
                                                              float(draft[field]), 0.01)
        c1, c2, c3 = st.columns(3)
        values["tempo_bpm"] = c1.number_input("Tempo (BPM)", 40, 220, int(draft["tempo_bpm"]))
        values["popularity"] = c2.number_input("Popularity (0-100)", 0, 100, int(draft["popularity"]))
        values["release_year"] = c3.number_input("Release year", 1900, 2026, int(draft["release_year"]))
        names = list(lib["playlists"])
        target = st.selectbox("Also add to playlist", ["(none)"] + names)
        if st.form_submit_button("Save song", type="primary"):
            if not genre.strip() or not mood.strip():
                st.error("Please fill in genre and mood.")
            else:
                song = add_song(lib, {**draft, **values, "genre": genre.strip().lower(),
                                      "mood": mood.strip().lower()})
                if target != "(none)":
                    add_to_playlist(lib, target, song["id"])
                save(lib)
                del st.session_state["draft"]
                flash(f"Saved '{song['title']}'")
                st.rerun()


# ---------------------------------------------------------------- page

def main() -> None:
    catalog = get_catalog()
    lib = load_library()
    if "flash" in st.session_state:
        st.toast(st.session_state.pop("flash"))

    with st.sidebar:
        st.header("🎵 Look")
        look = st.radio("Background", LOOKS, index=0,
                        help="Mood: the background color follows the mood of your latest "
                             "recommendation. Animated: it drifts and pulses to the top song's tempo.")
        st.toggle("Show the numbers", key="show_numbers",
                  help="Show raw scores and feature values for anyone curious.")
        st.caption(f"Catalog: {len(catalog)} songs · Your songs: {len(lib['songs'])} · "
                   f"Playlists: {len(lib['playlists'])}")

    st.title("Mood Music Recommender")
    st.caption("Gemini understands your request, a weighted scoring engine retrieves matching "
               "songs, and Gemini explains its picks using only those songs.")

    discover, playlists, add = st.tabs(["Discover", "My Playlists", "Add a Song"])
    with discover:
        discover_tab(catalog, lib)
    with playlists:
        playlists_tab(catalog, lib)
    with add:
        add_song_tab(catalog, lib)
    how_matching_works()

    # Applied last so it reflects mood changes made during this run (no one-click lag)
    apply_style(look, st.session_state.get("mood_prefs", {}), st.session_state.get("mood_bpm", 100))


main()
