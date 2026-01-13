# locations.py
from __future__ import annotations

import re
import json
from typing import Any, Callable, Dict, List, Optional

# -----------------------
# Small curated KB (expand anytime)
# -----------------------
# cost_tier: 1=budget, 2=mid, 3=premium, 4=very premium/long-haul
LOCATION_DB: Dict[str, Dict[str, Any]] = {
    "dubai": {
        "canonical": "Dubai",
        "country": "UAE",
        "region": "Middle East",
        "type": ["city", "international"],
        "cost_tier": 3,
        "visa_note": "Visa required (often straightforward via agent/e-visa depending on current policy).",
        "vibe": "shopping, modern city, desert safari, theme parks",
        "kid_friendly": True,
        "apr_note": "Warm to hot; plan indoor + evening activities.",
        "mar_note": "Pleasant to warm; good time for sightseeing and outdoor activities.",
    },
    "bali": {
        "canonical": "Bali",
        "country": "Indonesia",
        "region": "Southeast Asia",
        "type": ["island", "international"],
        "cost_tier": 3,
        "visa_note": "Visa rules vary; many Indians use e-visa/visa-on-arrival depending on current policy.",
        "vibe": "beaches, culture, cafes, relaxed villas",
        "kid_friendly": True,
        "apr_note": "Often good shoulder-season weather; check rainfall patterns.",
        "mar_note": "Often decent weather; still check rainfall patterns and humidity.",
    },
    "bhutan": {
        "canonical": "Bhutan",
        "country": "Bhutan",
        "region": "Himalayas",
        "type": ["mountains", "international", "short-haul"],
        "cost_tier": 2,
        "visa_note": "Indians often travel with permit/ID; confirm current rules before booking.",
        "vibe": "scenic, monasteries, calm, nature",
        "kid_friendly": True,
        "apr_note": "Spring is popular; pleasant weather.",
        "mar_note": "Spring is pleasant; good time to visit.",
    },
    "andaman": {
        "canonical": "Andaman & Nicobar Islands",
        "country": "India",
        "region": "Bay of Bengal",
        "type": ["islands", "domestic"],
        "cost_tier": 3,
        "visa_note": "Domestic travel; no visa needed.",
        "vibe": "beaches, clear water, snorkeling/scuba (optional)",
        "kid_friendly": True,
        "apr_note": "Generally good beach weather; humidity possible.",
        "mar_note": "Generally good beach weather; can be warm and humid.",
    },
    "ooty": {
        "canonical": "Ooty",
        "country": "India",
        "region": "Tamil Nadu (Nilgiris)",
        "type": ["hill station", "domestic"],
        "cost_tier": 2,
        "visa_note": "Domestic travel; no visa needed.",
        "vibe": "cooler hills, tea gardens, relaxed sightseeing",
        "kid_friendly": True,
        "apr_note": "Pleasant; can get crowded in holidays.",
        "mar_note": "Pleasant and not too cold; crowds vary.",
    },
    "manali": {
        "canonical": "Manali",
        "country": "India",
        "region": "Himachal Pradesh",
        "type": ["mountains", "domestic"],
        "cost_tier": 2,
        "visa_note": "Domestic travel; no visa needed.",
        "vibe": "mountains, scenic drives, light adventure",
        "kid_friendly": True,
        "apr_note": "Spring is nice; snow access varies by year.",
        "mar_note": "Cooler; snow access varies by year.",
    },
    "goa": {
        "canonical": "Goa",
        "country": "India",
        "region": "West Coast",
        "type": ["beach", "domestic"],
        "cost_tier": 2,
        "visa_note": "Domestic travel; no visa needed.",
        "vibe": "beaches, laid-back, food, short activities",
        "kid_friendly": True,
        "apr_note": "Hotter; choose beach + pool + relaxed schedule.",
        "mar_note": "Warm and pleasant; good for beach + pool days.",
    },
    "new zealand": {
        "canonical": "New Zealand",
        "country": "New Zealand",
        "region": "Oceania",
        "type": ["country", "international", "long-haul"],
        "cost_tier": 4,
        "visa_note": "Visa required; processing can take time. Plan early and keep documents ready.",
        "vibe": "nature, scenic drives, cities + outdoors, relaxed premium trip",
        "kid_friendly": True,
        "mar_note": "Late summer/early autumn vibe; weather varies by region—pack layers.",
        "apr_note": "Autumn begins; weather varies by region—pack layers.",
    },
    "shimla": {
        "canonical": "Shimla",
        "country": "India",
        "region": "Himachal Pradesh",
        "type": ["mountains", "domestic"],
        "cost_tier": 2,
        "visa_note": "Domestic travel; no visa needed.",
        "vibe": "hill station, relaxed sightseeing, mall road, viewpoints",
        "kid_friendly": True,
        "apr_note": "Pleasant spring weather; weekends can be busy.",
        "mar_note": "Cool to pleasant; good for relaxed sightseeing.",
    },
    "jaipur": {
        "canonical": "Jaipur",
        "country": "India",
        "region": "Rajasthan",
        "type": ["city", "domestic"],
        "cost_tier": 2,
        "visa_note": "Domestic travel; no visa needed.",
        "vibe": "forts, palaces, markets, culture, family-friendly sightseeing",
        "kid_friendly": True,
        "apr_note": "Warmer; plan early mornings and a relaxed midday break.",
        "mar_note": "Pleasant to warm; early starts help a lot.",
    },
    "islamabad": {
        "canonical": "Islamabad",
        "country": "Pakistan",
        "region": "South Asia",
        "type": ["city", "international", "short-haul"],
        "cost_tier": 3,
        "visa_note": "Visa required; check current requirements and processing timelines.",
        "vibe": "planned city, viewpoints, nearby nature",
        "kid_friendly": True,
        "apr_note": "Generally pleasant spring weather.",
        "mar_note": "Generally pleasant spring weather.",
    },
    "canada": {
        "canonical": "Canada",
        "country": "Canada",
        "region": "North America",
        "type": ["country", "international", "long-haul"],
        "cost_tier": 4,
        "visa_note": "Visa required; processing can take time.",
        "vibe": "cities + nature (varies by province)",
        "kid_friendly": True,
        "apr_note": "Weather varies widely; can be cold in many regions.",
        "mar_note": "Often cold in many regions; varies widely by province.",
    },
}

ALIASES: Dict[str, str] = {
    "dubai": "dubai",
    "bali": "bali",
    "bhutan": "bhutan",
    "andaman": "andaman",
    "andamans": "andaman",
    "andaman nicobar": "andaman",
    "ooty": "ooty",
    "manali": "manali",
    "goa": "goa",
    "canada": "canada",
    "shimla": "shimla",
    "jaipur": "jaipur",
    "islamabad": "islamabad",
    "new zealand": "new zealand",
}

_ALIAS_PAT = sorted(ALIASES.keys(), key=len, reverse=True)
_PLACES_RX = re.compile(r"\b(" + "|".join(re.escape(a) for a in _ALIAS_PAT) + r")\b", re.I)

_SUGGESTION_LEADIN = re.compile(
    r"\b(recommend|suggest|consider|go to|visit|try|opt for|better option)\b",
    re.I,
)

_AFTER_SUGGESTION_RX = re.compile(
    r"\b(?:recommend|suggest|consider|visit|try|opt for|better option)\s+([a-zA-Z][a-zA-Z\-']+(?:\s+[a-zA-Z][a-zA-Z\-']+){0,3})",
    re.I,
)

DYNAMIC_DB: Dict[str, Dict[str, Any]] = {}

_STOP = {
    "a","an","the","maybe","some","any","good","great","nice","very","best",
    "option","destination","destinations","location","locations","premium","popular",
    "like","such","as","kind","types"
}

def normalize_place(name: str) -> str:
    s = re.sub(r"\s+", " ", (name or "").strip().lower())
    s = re.sub(r"[^\w\s&\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_domestic(card: Dict[str, Any]) -> bool:
    c = (card.get("country") or "").strip().lower()
    t = [x.lower() for x in (card.get("type") or []) if isinstance(x, str)]
    return (c == "india") or ("domestic" in t)

def _clean_phrase_to_key(phrase: str) -> str:
    phrase = normalize_place(phrase)
    toks = [w for w in phrase.split() if w not in _STOP]
    return " ".join(toks[:4]).strip()

def llm_extract_places(agent_text: str, llm_generate: Callable[[str], str], max_places: int) -> List[str]:
    prompt = f"""
Return ONLY strict JSON:
{{"places":[string]}}

Task: Extract destination place names mentioned/suggested in this agent message.
Rules:
- Return ONLY real place names (cities/regions/countries/islands).
- Ignore generic phrases like "premium locations", "good option", "hill station".
- If multiple destinations are mentioned, include up to {max_places}.
- Preserve proper names like "New Zealand", "United Kingdom", "Beijing", "Australia".

AGENT_TEXT:
{agent_text}
""".strip()

    raw = llm_generate(prompt) or ""
    try:
        data = json.loads(raw)
        places = data.get("places", [])
        if not isinstance(places, list):
            return []
        out: List[str] = []
        for p in places:
            if not isinstance(p, str):
                continue
            key = _clean_phrase_to_key(p)
            if not key:
                continue
            key = ALIASES.get(key, key)
            if key not in out:
                out.append(key)
            if len(out) >= max_places:
                break
        return out
    except Exception:
        return []

def extract_places(agent_text: str, max_places: int = 3, llm_generate: Optional[Callable[[str], str]] = None) -> List[str]:
    t = agent_text or ""
    found: List[str] = []

    # 1) alias table
    for m in _PLACES_RX.finditer(t):
        key = normalize_place(m.group(1))
        canon = ALIASES.get(key, key)
        if canon not in found:
            found.append(canon)
        if len(found) >= max_places:
            return found

    # 2) heuristic capture after recommend/suggest/etc
    if _SUGGESTION_LEADIN.search(t):
        m = _AFTER_SUGGESTION_RX.search(t)
        if m:
            phrase = _clean_phrase_to_key(m.group(1))
            if phrase:
                canon = ALIASES.get(phrase, phrase)
                if canon and canon not in found:
                    found.append(canon)
                    if len(found) >= max_places:
                        return found

    # 3) LLM fallback
    if not found and llm_generate is not None and len(t.split()) >= 3:
        found = llm_extract_places(t, llm_generate, max_places=max_places)

    return found[:max_places]

def get_place_card(place_key: str, llm_generate: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    pk = normalize_place(place_key)

    if pk in LOCATION_DB:
        return LOCATION_DB[pk]
    if pk in DYNAMIC_DB:
        return DYNAMIC_DB[pk]

    # If no LLM available, return a safe conservative default
    if llm_generate is None:
        card = {
            "canonical": place_key.title(),
            "country": "",
            "region": "",
            "type": ["unknown"],
            "cost_tier": 3,
            "visa_note": "",
            "vibe": "",
            "kid_friendly": True,
            "apr_note": "",
            "mar_note": "",
            "unknown": True,
        }
        DYNAMIC_DB[pk] = card
        return card

    prompt = f"""
Return ONLY strict JSON with these keys:
{{
  "canonical": string,
  "country": string,
  "region": string,
  "type": [string],
  "cost_tier": 1|2|3|4,
  "visa_note": string,
  "vibe": string,
  "kid_friendly": boolean,
  "mar_note": string,
  "apr_note": string
}}

Rules:
- Keep it general and non-controversial.
- No exact prices, no guarantees.
- If the place is in India, include type "domestic" and say "no visa needed".
- If international, include type "international" and give a generic visa note.
- cost_tier guidance:
  - nearby short-haul (Nepal/Bhutan etc): 2
  - common international city trips (Dubai, Singapore): 3
  - long-haul (US/UK/Europe/Australia/NZ): 4

PLACE: {place_key}
""".strip()

    raw = llm_generate(prompt) or ""
    try:
        card = json.loads(raw)
    except Exception:
        card = {}

    if not isinstance(card, dict):
        card = {}

    card.setdefault("canonical", place_key.title())
    card.setdefault("country", "")
    card.setdefault("region", "")
    card.setdefault("type", ["international"])
    card.setdefault("cost_tier", 3)
    card.setdefault("visa_note", "")
    card.setdefault("vibe", "")
    card.setdefault("kid_friendly", True)
    card.setdefault("mar_note", "")
    card.setdefault("apr_note", "")

    DYNAMIC_DB[pk] = card
    return card

# -----------------------
# UPDATED: reaction now avoids repeating kids/visa/weather for same place
# -----------------------
def build_place_reaction(
    place_cards: List[Dict[str, Any]],
    *,
    has_kids: bool,
    budget_tier: int,
    already_asked_weather: bool,
    already_asked_kids: bool,
    already_asked_visa: bool,
    travel_month: str,
) -> str:
    if not place_cards:
        return ""

    c0 = place_cards[0]
    name = c0.get("canonical", "That place")
    country = (c0.get("country") or "").strip()
    cost_tier = int(c0.get("cost_tier", 3))
    visa_note = (c0.get("visa_note") or "").strip()

    where = f" in {country}" if country else ""
    domestic = is_domestic(c0)

    if cost_tier >= 4 and budget_tier <= 3:
        fit = "I’m a bit worried it might go over budget."
    elif cost_tier >= 3 and budget_tier <= 2:
        fit = "It might be slightly on the pricey side for us."
    else:
        fit = "It could work in our budget."

    kid_bit = ""
    if has_kids and (not already_asked_kids):
        kid_bit = " Is it easy with kids?"

    visa_bit = ""
    if (not domestic) and visa_note and re.search(r"\bvisa|permit\b", visa_note, re.I) and (not already_asked_visa):
        visa_bit = " Also, any visa/processing time we should plan for?"

    month = (travel_month or "").strip().lower()
    month_note = ""
    if month.startswith("apr"):
        month_note = (c0.get("apr_note") or "").strip()
    elif month.startswith("mar"):
        month_note = (c0.get("mar_note") or "").strip()

    weather_bit = ""
    if (not already_asked_weather) and month_note:
        weather_bit = f" And how is it in {travel_month.strip()}?"

    return f"{name}{where} sounds interesting—{fit}{kid_bit}{visa_bit}{weather_bit}".strip()
