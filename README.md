# PropertyScout AI — Streamlit Edition

A memory-aware, tool-using real estate advisor agent — now with a chat UI,
powered by **Groq** (instead of Gemini) and deployable on **Streamlit
Community Cloud**.

This is a port of the original Kaggle × Google AI Agents Intensive capstone
notebook (`Day5_PropertyScout_FINAL.ipynb`) into a standalone Streamlit app.

## What changed from the original notebook

| | Original (Colab) | This version |
|---|---|---|
| LLM | Gemini API (`gemini-2.5-flash`) | Groq API (`llama-3.3-70b-versatile`) |
| Live search | Gemini's built-in Google Search grounding | `ddgs` (DuckDuckGo search, free, no key) |
| Memory storage | Google Drive JSON file | Local JSON file (`data/user_profile.json`) |
| Interface | Jupyter notebook | Streamlit chat UI |

The **three tools** (`search_listings`, `estimate_price`,
`update_user_profile`), the **guardrail rules**, and the **system prompt
logic** are carried over as-is from the original capstone.

## Project structure

```
propertyscout-streamlit/
├── app.py              # Streamlit UI
├── agent.py            # System prompt, guardrails, Groq tool-calling loop
├── tools.py             # search_listings, estimate_price, update_user_profile
├── requirements.txt
├── .streamlit/
│   └── secrets.toml.example
├── .gitignore
└── data/                 # created automatically — stores user_profile.json
```

## Run locally

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your_groq_api_key_here"   # Windows: set GROQ_API_KEY=...
streamlit run app.py
```

Get a free Groq API key at **https://console.groq.com/keys**.

## Deploy on Streamlit Community Cloud

1. Push this folder to a **public or private GitHub repo**.
2. Go to **https://share.streamlit.io** → **New app**.
3. Select your repo/branch, set **Main file path** to `app.py`.
4. Before (or after) deploying, go to **App settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
5. Click **Deploy**. That's it — no other config needed.

> ⚠️ Never commit your real `GROQ_API_KEY` or a real `.streamlit/secrets.toml`
> to GitHub. Only `secrets.toml.example` (a placeholder) is included here.

## Limitations & notes (carried over + updated)

- `estimate_price()` is a simple city-tier heuristic, not a trained ML model.
- `search_listings()` now uses free DuckDuckGo search instead of Gemini's
  grounding — results are less structured, so the agent is instructed to
  still caveat any numbers as unverified.
- **Memory persistence**: Streamlit Community Cloud's filesystem is
  ephemeral — it can reset on redeploys or when the app goes to sleep and
  wakes up. This still gives cross-session memory during normal use (same
  mechanism as before, just local disk instead of Drive), but for guaranteed
  long-term persistence you'd want to swap `tools.py`'s `load_profile` /
  `save_profile` for a small hosted database.
- Groq's free tier has generous but finite rate limits — heavy testing in a
  short window may hit them.

## Original capstone

The original Gemini/Colab notebook, writeup, and Kaggle submission this was
ported from are unchanged and still available as
`Day5_PropertyScout_FINAL.ipynb` / `PropertyScout_AI_Capstone_Writeup.docx`.
