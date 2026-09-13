"""
tools.py — PropertyScout AI's three agent tools.

Ported from the original Gemini/Colab capstone notebook to work standalone
with Groq's OpenAI-compatible function-calling API.

Key change from the original:
- search_listings originally used Gemini's built-in Google Search grounding
  tool. Groq has no equivalent built-in web search, so this version uses the
  free `duckduckgo-search` (ddgs) package instead — no extra API key needed.
"""

import os
import json
from ddgs import DDGS

# ---------------------------------------------------------------------------
# Persistent memory (user profile)
# ---------------------------------------------------------------------------
# NOTE: Streamlit Community Cloud's filesystem is ephemeral — it can reset on
# redeploys or app sleep/wake cycles. This still gives cross-session memory
# during normal usage (same as writing to disk did in Colab), but it is not
# guaranteed to survive forever the way Google Drive did. For guaranteed
# persistence you'd swap this for a small hosted DB (e.g. Supabase, a Google
# Sheet, or Streamlit's own st.connection to a database).
MEMORY_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(MEMORY_DIR, exist_ok=True)
MEMORY_FILE = os.path.join(MEMORY_DIR, "user_profile.json")

DEFAULT_PROFILE = {
    "preferred_city": None,
    "budget_pkr": None,
    "property_type": None,
    "notes": [],
}


def load_profile() -> dict:
    """Loads the user profile from disk, or returns a blank default if none exists yet."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return DEFAULT_PROFILE.copy()


def save_profile(profile: dict) -> None:
    """Saves the user profile to disk so it survives across sessions."""
    with open(MEMORY_FILE, "w") as f:
        json.dump(profile, f, indent=2)


# ---------------------------------------------------------------------------
# Tool 1: search_listings (live web search via DuckDuckGo, cached in-memory)
# ---------------------------------------------------------------------------
_search_cache = {}


def search_listings(city: str, property_type: str = "residential plot", budget_pkr: int = None) -> str:
    """Searches the web for current real estate listings/prices in a given city.

    Args:
        city: The city or area to search in (e.g. 'DHA Multan', 'Bahria Town Lahore').
        property_type: Type of property (e.g. 'residential plot', 'house', 'apartment').
        budget_pkr: Optional budget in PKR (or local currency) to narrow results.

    Returns:
        A text summary of what was found on the web about current listings/prices.
    """
    cache_key = (city.lower(), property_type.lower(), budget_pkr)
    if cache_key in _search_cache:
        return _search_cache[cache_key]

    query = f"current {property_type} prices and listings in {city}"
    if budget_pkr:
        query += f" under {budget_pkr} PKR"

    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=5))
    except Exception as e:
        return (f"(Live search failed: {e}). No web data available right now — "
                f"treat any number as unverified and fall back to a heuristic estimate.")

    if not hits:
        result = "No relevant live listings found for this search. Try estimate_price instead for a rough number."
    else:
        lines = [f"Web search results for: {query}\n"]
        for h in hits:
            title = h.get("title", "").strip()
            body = h.get("body", "").strip()
            href = h.get("href", "").strip()
            lines.append(f"- {title}: {body} ({href})")
        result = "\n".join(lines)

    _search_cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Tool 2: estimate_price (simple, transparent city-tier heuristic)
# ---------------------------------------------------------------------------
# NOT a trained ML model — intentionally simple. See README "Limitations".
CITY_TIER_RATES = {
    "tier_1": {"min": 1_500_000, "max": 4_000_000},   # Lahore, Karachi, Islamabad
    "tier_2": {"min": 800_000,   "max": 2_000_000},   # Multan, Faisalabad, Rawalpindi, Peshawar
    "tier_3": {"min": 300_000,   "max": 900_000},     # smaller cities/towns
}
TIER_1_CITIES = ["lahore", "karachi", "islamabad"]
TIER_2_CITIES = ["multan", "faisalabad", "rawalpindi", "peshawar"]


def estimate_price(city: str, area_marla: float, property_type: str = "residential plot") -> str:
    """Gives a rough heuristic price estimate based on city tier and size.

    Args:
        city: The city name (e.g. 'Multan', 'Lahore').
        area_marla: Size of the property in Marla (1 Marla = 225 sq ft).
        property_type: Type of property; commercial gets a higher multiplier.

    Returns:
        A text explanation of the estimated price range, clearly labeled as a rough heuristic.
    """
    city_lower = city.lower()
    if any(c in city_lower for c in TIER_1_CITIES):
        tier = "tier_1"
    elif any(c in city_lower for c in TIER_2_CITIES):
        tier = "tier_2"
    else:
        tier = "tier_3"

    rates = CITY_TIER_RATES[tier]
    multiplier = 1.5 if "commercial" in property_type.lower() else 1.0
    low = int(rates["min"] * area_marla * multiplier)
    high = int(rates["max"] * area_marla * multiplier)

    return (f"ROUGH heuristic estimate (not live market data) for a {area_marla} marla {property_type} "
            f"in {city}: PKR {low:,} to PKR {high:,}. This uses a simplified city-tier model "
            f"({tier.replace('_', ' ')}) and should be cross-checked against live listings.")


# ---------------------------------------------------------------------------
# Tool 3: update_user_profile (persistent memory writer)
# ---------------------------------------------------------------------------
def update_user_profile(preferred_city: str = None, budget_pkr: int = None,
                         property_type: str = None, new_note: str = None) -> str:
    """Updates the persistent user profile with new information learned during the conversation.
    Call this whenever the user shares a preference worth remembering for future sessions.

    Args:
        preferred_city: City the user is interested in, if mentioned.
        budget_pkr: User's budget in PKR, if mentioned.
        property_type: Type of property they're interested in, if mentioned.
        new_note: Any other useful preference or fact worth remembering, as a short string.

    Returns:
        Confirmation of what was saved.
    """
    profile = load_profile()
    if preferred_city:
        profile["preferred_city"] = preferred_city
    if budget_pkr:
        profile["budget_pkr"] = budget_pkr
    if property_type:
        profile["property_type"] = property_type
    if new_note:
        profile["notes"].append(new_note)
    save_profile(profile)
    return f"Profile updated and saved: {profile}"


# ---------------------------------------------------------------------------
# Tool schema for Groq's OpenAI-compatible function-calling API
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_listings",
            "description": "Searches the web for current real estate listings/prices in a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city or area to search in."},
                    "property_type": {"type": "string", "description": "e.g. 'residential plot', 'house', 'apartment', 'commercial'."},
                    "budget_pkr": {"type": "integer", "description": "Optional budget in PKR to narrow results."},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_price",
            "description": "Gives a rough heuristic price estimate based on city tier and size, for when live search isn't needed or available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name."},
                    "area_marla": {"type": "number", "description": "Size of the property in Marla (1 Marla = 225 sq ft)."},
                    "property_type": {"type": "string", "description": "Type of property; commercial gets a higher multiplier."},
                },
                "required": ["city", "area_marla"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_user_profile",
            "description": "Saves a new user preference (city, budget, property type, or a note) to persistent memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "preferred_city": {"type": "string"},
                    "budget_pkr": {"type": "integer"},
                    "property_type": {"type": "string"},
                    "new_note": {"type": "string"},
                },
                "required": [],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "search_listings": search_listings,
    "estimate_price": estimate_price,
    "update_user_profile": update_user_profile,
}
