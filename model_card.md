# 🎧 Model Card: Vibenator 6000 (AI Music Recommender)

## 1. Model Name  

**Vibenator 6000**: it hunts down songs that fit the vibe you describe.

It's a system rather than a single model. It has two parts:
- **Google Gemini**, a large language model. It reads requests, explains picks and estimates song features. The default is `gemini-3.8-flash`; the final samples and evaluation ran on `gemini-3.5-flash-lite`.
- **A hand-tuned scoring engine** that I built in Module 3. It decides which songs are retrieved.

---

## 2. Intended Use  

**Goal / Task:** You describe what you want to hear in plain English, or pick a playlist you've built. The Vibenator 6000 recommends songs from its catalog and your own saved songs, and explains each pick in plain English. You can also add your own songs, and Gemini estimates their features for you to review.

**Assumptions:**
- Your taste can be summed up by a genre, a mood, and a few 0–1 values: energy, acousticness, valence, instrumentalness and discovery.
- A playlist's taste is the average of its songs.
- Gemini can translate everyday language into those values.
- It knows nothing about your listening history.

**Intended use:**
- A portfolio and classroom project showing retrieval-augmented generation (RAG) with guardrails, fallbacks and testing.
- Exploring how an LLM and a transparent scoring system can work together.
- Showing how weights, data, and an AI's guesses create bias.

**Not intended for:**
- Real users or a real music app. The catalog has only 20 made-up songs.
- Decisions about artists, such as promotion or payment. Song features are hand-written or estimated by an AI, not measured.
- Judging anyone's "real" music taste.

---

## 3. How the Model Works  

**The pipeline for a text request:**
1. **Understand.** Gemini turns your words into a taste profile. For example, "music to get hyped to at the gym" became *edm, intense, energy 0.95*.
2. **Guardrail.** The profile is checked before it's used: numbers are clamped to 0–1, genres and moods that aren't in the catalog are dropped, and missing values get defaults.
3. **Retrieve.** The scoring engine scores every song against the profile and keeps the top 5.
4. **Generate.** Gemini picks up to 3 of those 5 and explains why, using only the retrieved songs and their score details.
5. **Verify.** Any song Gemini names that wasn't retrieved is thrown away. If Gemini fails, or none of its picks survive, you get the top-ranked songs instead, with a note explaining why.

**The scoring engine** (from Module 3) is a points game:
- **Genre match:** +2 points.
- **Mood match:** +1 point.
- **Energy:** up to 1.5 points, for being *close* to your target (not for being "more" energetic).
- **Acousticness:** up to 0.75 points, closeness again.
- **Valence (happiness) and instrumentalness:** up to 0.5 points each, closeness again.
- **Discovery:** up to 0.5 points, a small boost toward hidden gems or toward hits.

The best possible score is 6.75. In the app it's shown as a **match %**, with reasons such as "✓ Same genre (rock)" or "Energy: very close to what you want".

**Playlists:** a playlist becomes a taste profile by averaging its songs' features and taking the most common genre and mood. A tie means no genre or mood preference. Discovery comes from how popular the songs are. Songs already in the playlist are never recommended back. An optional tweak, like "but more acoustic", is parsed by Gemini and blended 50/50 with the playlist profile.

**Adding your own song:** Gemini estimates its genre, mood, energy, tempo, popularity and other features. The values are clamped to valid ranges, songs Gemini doesn't recognize are flagged, and **you review and can edit every value before saving.**

---

## 4. Data  

- **Catalog:** 20 fictional songs in `data/songs.csv`. The starter had 10, and I added 10 more.
  - **Genres (17):** pop, indie pop, lofi, rock, ambient, jazz, synthwave, hip hop, classical, country, edm, r&b, metal, reggae, folk, latin, blues.
  - **Moods (16):** including happy, chill, intense, sad, angry, romantic, nostalgic, euphoric and peaceful.
  - **Features per song:** genre, mood, energy, tempo, valence, danceability, acousticness, instrumentalness, popularity (0–100) and release year.
- **User songs and playlists:** saved locally in `data/user_library.json`. That file is never committed. User songs are marked as AI-estimated or entered by hand.
- **Gemini:** a general-purpose model with no fine-tuning. Prompts include the catalog's genre and mood lists so its answers fit the scoring engine.

**Limits:**
- The catalog is tiny and uneven: most genres have 1 song, while lofi has 3.
- Catalog values are made up, and user-song values are AI guesses. Neither is a real audio measurement. Spotify's audio-features API is closed to new apps.
- Nothing about lyrics, language, culture or listening history.

---

## 5. Strengths  

- **Grounded answers.** Gemini can only recommend songs the retriever found. In the evaluation, 6 of 6 requests were fully grounded, with no invented songs.
- **Understands everyday language.** "Sad acoustic songs for a rainy day" and "music to get hyped to at the gym" produced opposite, sensible profiles (energy 0.2 vs. 0.95) and fitting picks.
- **Explainable.** Every pick shows a match % and plain-English reasons, and a toggle reveals the exact points per feature.
- **Always gives an answer.** With no API key, an overloaded API or a used-up quota, the app falls back to the ranked list and says why. Every step is logged.
- **A human checks the AI's guesses.** AI-estimated song features are shown for review before they're saved.

---

## 6. Limitations and Bias 

**From the AI side:**
- **Estimated features can be wrong.** For a song Gemini doesn't recognize, the values are guesses, and they directly affect recommendations. The review step and the "doesn't recognize this song" warning reduce this risk but don't remove it.
- **Gemini's own judgment shapes the profile.** In testing, it mapped "k-pop" to the catalog genre "pop" rather than leaving the genre blank. That's reasonable, but it means the AI's assumptions about how genres relate become part of the result.
- **Availability.** The AI features depend on Gemini's free tier: 20 requests per day for some models and 15 per minute. When limits are hit, recommendations fall back to the less personal ranked list.

**From the scoring engine** (found in Module 3; full details in [docs/original-project.md](docs/original-project.md)):
- **An exact genre + mood match can overpower everything else.** Those two labels are worth 3 points together, while a song that's completely wrong on energy loses only about 0.8. In the "Sad but Hyped" test, a user asking for energy 0.9 got a quiet blues song first.
- **Exact text matching.** "Intense" and "angry", or "pop" and "indie pop", count as unrelated. This creates a filter bubble around the user's own labels.
- **Discovery is weak.** The popularity bonus is at most 0.5 points, so it rarely changes the order.

**From playlists:**
- **Averaging blurs mixed taste.** A playlist that's half metal and half lofi becomes a "medium" profile that fits neither half.

---

## 7. Evaluation  

- **Unit tests (26 pass).** They use a fake LLM, so they need no API key and give the same result every time. They check:
  - grounding, and that invented songs are dropped
  - the fallback on API errors and on a used-up daily quota
  - input validation and clamping
  - playlist profiles, tie handling and duplicate songs
  - recovery from a corrupted library file
  - that playlist songs are never recommended back
  - blending a tweak with a playlist
  - plain-English explanations
  - the original Module 3 scoring rules
- **Web app checks.** I used Streamlit's headless test tool to script:
  - switching playlists with one click
  - the background updating immediately
  - adding a song with one click
  - delete confirmation
  - the numbers toggle
- **Live evaluation (`src/evaluate.py`, real Gemini).** It sent 6 fixed requests, plus 2 repeats for consistency.
  - **Results:** parsing 5/6, grounding 6/6, AI used 6/6, consistency 2/2.
  - **The one failure:** the k-pop request hit the per-minute rate limit. Rerun a minute later, Gemini got the energy right but chose "pop" as the genre.
- **Module 3 profile tests.** The scoring engine was stress-tested with 3 normal and 4 adversarial profiles, plus a weight-shift experiment. The results and comparisons are in [docs/original-project.md](docs/original-project.md).

**Does it match my intuition?** Mostly, yes. The gym request returned EDM, pop and rock songs with energy above 0.9. The rainy-day request returned melancholic folk, blues and quiet classical. When Gemini was unavailable, the fallback was honest about it, but less personal: for example, it ignored a "but with vocals" tweak.

---

## 8. Future Work  

1. **Partial credit for similar labels.** "Pop" and "indie pop", or "chill" and "peaceful", should earn some points. Gemini could even suggest these similarities, as long as a guardrail checks them.
2. **A real feature source.** Measure audio features from user-uploaded audio files instead of relying on AI guesses. The same audio could also drive the mood background in real time.
3. **Better playlist profiles.** Detect mixed playlists and recommend for each cluster separately, instead of averaging them into one profile.
4. **Smarter rate-limit handling.** Pause between requests, cache repeated answers, and wait longer when the per-minute limit is hit.
5. **Rebalance the weights and add songs.** More energy weight and less genre weight gave better results in my Module 3 experiment, and a larger catalog would give every genre real options.

---

## 9. Personal Reflection  

This project taught me that AI truly isn't perfect, and that testing the app yourself and confirming the changes your AI made still has real value. I found myself going back and forth to see if it actually did what I asked it to do because features it included looked fine at first.
The better your initial set up is, the easier it is to continue working. I even had a small moment where I needed to refactor because of a major feature that I added (for example changing all the weight numbers to something easily readable).
I also got to see additional features of Claude, how it prompts the user in multi-steps on what the user wants, which was helpful and often times I had to add on top of what Claude wanted to do.
I've learned how RAG can be implemented in another system.
I've learned how a connected API can be overloaded.
I even realized that features I wanted weren't so great until they were placed alongside everything else.

**A suggestion I accepted:** When I needed an AI feature, Claude suggested building RAG on top of my original Module 3 scoring engine instead of replacing it. Gemini reads my request, my scorer retrieves the top 5 songs, and Gemini is only allowed to pick from those 5. Any song it invents gets thrown out. I liked this because my weighted scoring didn't go to waste; it became the part that keeps the AI honest. Later, I also accepted its fix for the API quota: the app had been retrying errors that were actually a daily limit, so it was wasting requests and time.

**A suggestion I rejected or revised:** Claude suggested adding about 40 more made-up songs to the catalog. Instead, I wanted users to make their own playlists and add their own songs, with the taste profile learned from the playlist. That turned into a bigger and better feature than just more data. I also revised a lot of the UI after testing it myself: the raw score numbers were confusing, so I asked for plain-English reasons and a match % with a "Show the numbers" toggle. The playlist list needed two clicks to switch, so I asked for a dropdown. I also asked for a confirmation before deleting a playlist.

**Limitations of my system:** The catalog is only 20 fictional songs, so some tastes don't have many good matches. When a user adds a song, Gemini guesses its features, and those guesses can be wrong, especially for songs it doesn't know. That's why the user has to review them before saving. A playlist's taste is an average, so a mixed playlist (half metal, half lofi) ends up as a "medium" profile that fits neither half. Genres and moods only match exactly, so "chill" and "relaxed" count as different. The AI also depends on Gemini's free tier, which only allows 20 requests a day for some models; when that runs out, the app falls back to a less personal ranked list.
