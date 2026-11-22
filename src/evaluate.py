"""
Reliability evaluation for the AI recommender (uses the real Gemini API).

Runs a fixed set of requests and checks:
  - parsing:     the LLM's preferences point in the expected direction
  - grounding:   every recommended song exists in the retrieved set (no hallucinations)
  - AI success:  the AI path was used rather than the fallback
  - consistency: running the same request twice gives the same top pick

Run from the project root with:  python -m src.evaluate
"""

from dotenv import load_dotenv

from src.ai_recommender import GeminiClient, log, recommend_from_text
from src.recommender import load_songs

# Each case: request, plus checks on the parsed prefs (key -> (min, max) or exact string)
CASES = [
    ("calm lofi beats to study to, no vocals",
     {"favorite_genre": "lofi", "target_energy": (0.0, 0.5), "target_instrumentalness": (0.6, 1.0)}),
    ("loud aggressive music for lifting weights",
     {"target_energy": (0.75, 1.0)}),
    ("sad acoustic songs for a rainy day",
     {"target_valence": (0.0, 0.4), "target_acousticness": (0.6, 1.0)}),
    ("happy upbeat pop for a road trip",
     {"favorite_genre": "pop", "target_valence": (0.6, 1.0), "target_energy": (0.6, 1.0)}),
    ("obscure underground tracks nobody has heard of",
     {"discovery": (0.7, 1.0)}),
    ("k-pop for a dance party",  # genre not in catalog: guardrail should blank it
     {"favorite_genre": "", "target_energy": (0.6, 1.0)}),
]


def check_prefs(prefs: dict, expected: dict) -> list:
    """Return a list of failure messages (empty list = all checks passed)."""
    failures = []
    for key, rule in expected.items():
        value = prefs.get(key)
        if isinstance(rule, tuple):
            low, high = rule
            if not low <= value <= high:
                failures.append(f"{key}={value} not in [{low}, {high}]")
        elif value != rule:
            failures.append(f"{key}={value!r}, expected {rule!r}")
    return failures


def main() -> None:
    load_dotenv()
    songs = load_songs("data/songs.csv")
    client = GeminiClient()
    totals = {"parsing": 0, "grounding": 0, "ai_used": 0, "consistency": 0}

    for request, expected in CASES:
        first = recommend_from_text(request, songs, client=client)
        second = recommend_from_text(request, songs, client=client)

        retrieved = {s["title"] for s, _, _ in first["retrieved"]}
        results = {
            "parsing": not check_prefs(first["prefs"], expected),
            "grounding": all(p["song"]["title"] in retrieved for p in first["picks"]),
            "ai_used": first["used_ai"],
            "consistency": first["picks"][0]["song"]["title"] == second["picks"][0]["song"]["title"],
        }
        for key, ok in results.items():
            totals[key] += ok

        status = "PASS" if all(results.values()) else "FAIL"
        print(f"[{status}] {request}")
        print(f"       top pick: {first['picks'][0]['song']['title']}")
        for msg in check_prefs(first["prefs"], expected):
            print(f"       parsing: {msg}")
        for key, ok in results.items():
            if not ok and key != "parsing":
                print(f"       {key}: failed")

    n = len(CASES)
    print("\nSummary")
    for key, count in totals.items():
        print(f"  {key:<12} {count}/{n}  ({count / n:.0%})")
    log.info("evaluation summary: %s of %d", totals, n)


if __name__ == "__main__":
    main()
