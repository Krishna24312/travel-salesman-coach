# main.py
# Run:
#   python -m venv .venv
#   source .venv/bin/activate
#   pip install -r requirements.txt
#   uvicorn main:app --reload --port 8000
#
# NOTE: If you use --reload, sessions reset in RAM on every code save.
# This version persists sessions/reports to disk so /report works even after reload.

import os
import uuid
import re
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

# Twilio router (make this optional so local runs don't break if twilio isn't installed yet)
try:
    from twilio_phone import router as twilio_router  # type: ignore
except Exception:
    twilio_router = None  # type: ignore

# -----------------------
# Rate limiting (SlowAPI)
# -----------------------
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.responses import PlainTextResponse

# location knowledge
from locations import extract_places, get_place_card, build_place_reaction, normalize_place

# Optional: Google GenAI SDK
try:
    from google import genai  # type: ignore
except Exception:
    genai = None  # type: ignore

load_dotenv()

# -----------------------
# Logging
# -----------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s %(asctime)s %(message)s")
log = logging.getLogger("client_sim")

# -----------------------
# App setup
# -----------------------
app = FastAPI()

if twilio_router is not None:
    app.include_router(twilio_router)
else:
    log.warning("Twilio router not loaded (twilio_phone.py or twilio package missing). Twilio endpoints disabled.")

# -----------------------
# Rate limiting (SlowAPI)
# -----------------------
def _client_ip(request: Request) -> str:
    """
    Prefer X-Forwarded-For (Render/proxies) else fall back to request.client.host.
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        ip = xff.split(",")[0].strip()
        if ip:
            return ip
    return get_remote_address(request)


limiter = Limiter(key_func=_client_ip)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return PlainTextResponse("Too many requests. Please slow down.", status_code=429)


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -----------------------
# Persistence (fix report Not Found after reload)
# -----------------------
DATA_DIR = Path(os.getenv("DATA_DIR", ".data")).resolve()
SESS_DIR = DATA_DIR / "sessions"
SESS_DIR.mkdir(parents=True, exist_ok=True)


def _session_path(session_id: str) -> Path:
    return SESS_DIR / f"{session_id}.json"


def _json_safe(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def save_session(session: Dict[str, Any]) -> None:
    sid = (session.get("session_id") or "").strip()
    if not sid:
        return
    scenario_id = session.get("scenario_id")
    state = session.get("state", {})
    transcript = session.get("transcript", [])
    report = session.get("report", None)
    payload = {
        "session_id": sid,
        "turn": int(session.get("turn", 0)),
        "done": bool(session.get("done", False)),
        "scenario_id": scenario_id,
        "state": _json_safe(state),
        "transcript": _json_safe(transcript),
        "report": _json_safe(report),
    }
    try:
        _session_path(sid).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning("Failed to persist session %s: %s", sid, e)


def load_session_from_disk(session_id: str) -> Optional[Dict[str, Any]]:
    p = _session_path(session_id)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None


SESSIONS: Dict[str, Dict[str, Any]] = {}

# -----------------------
# Deepgram STT settings
# -----------------------
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_LISTEN_URL = "https://api.deepgram.com/v1/listen"
DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-3")
DEEPGRAM_LANG = os.getenv("DEEPGRAM_LANG", "en-IN")


def _deepgram_request(audio_bytes: bytes, content_type: str) -> Dict[str, Any]:
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY not set")

    params = {
        "model": DEEPGRAM_MODEL,
        "language": DEEPGRAM_LANG,
        "smart_format": "true",
        "punctuate": "true",
    }
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": content_type or "application/octet-stream",
    }

    r = requests.post(
        DEEPGRAM_LISTEN_URL,
        params=params,
        headers=headers,
        data=audio_bytes,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def deepgram_extract_text(dg_json: Dict[str, Any]) -> Tuple[str, float]:
    try:
        channels = (dg_json.get("results", {}) or {}).get("channels", []) or []
        alternatives = (channels[0] or {}).get("alternatives", []) or []
        alt0 = alternatives[0] if alternatives else {}
        text = (alt0.get("transcript") or "").strip()
        conf = float(alt0.get("confidence") or 0.0)
        return text, conf
    except Exception:
        return "", 0.0


@app.post("/api/stt")
@limiter.limit("10/minute")
async def api_stt(request: Request, audio: UploadFile = File(...)):
    if not DEEPGRAM_API_KEY:
        return JSONResponse({"error": "DEEPGRAM_API_KEY not set"}, status_code=500)

    audio_bytes = await audio.read()
    if not audio_bytes:
        return JSONResponse({"error": "Empty audio upload"}, status_code=400)

    try:
        dg = await run_in_threadpool(
            _deepgram_request,
            audio_bytes,
            audio.content_type or "audio/webm",
        )
        text, conf = deepgram_extract_text(dg)
        return {"text": text, "confidence": conf}
    except Exception as e:
        return JSONResponse({"error": f"Deepgram STT failed: {e}"}, status_code=500)


# -----------------------
# LLM Providers (Groq primary, Gemini fallback)
# -----------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-8b-instant").strip()
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")

GEMINI_CLIENT_MODEL = os.getenv("GEMINI_CLIENT_MODEL", "gemini-2.5-flash")
GEMINI_REPAIR_MODEL = os.getenv("GEMINI_REPAIR_MODEL", "gemini-2.5-flash-lite")
GEMINI_API_KEY = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()

_gemini_client = None
if GEMINI_API_KEY and genai is not None:
    try:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        log.info("Gemini client enabled.")
    except Exception as e:
        _gemini_client = None
        log.warning("Gemini client init failed (will disable Gemini): %s", e)


def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except Exception:
        return ""


def groq_generate_text(
    prompt: str,
    model: str = "",
    temperature: float = 0.2,
    max_output_tokens: int = 300,
    timeout: int = 25,
) -> str:
    if not GROQ_API_KEY:
        return ""

    model = (model or GROQ_MODEL).strip()
    url = f"{GROQ_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(max_output_tokens),
    }

    try:
        r = requests.post(url, headers=headers, json=body, timeout=timeout)
        if r.status_code >= 400:
            log.warning("Groq error %s: %s", r.status_code, r.text[:220])
            return ""
        data = r.json()
        txt = ((((data.get("choices") or [{}])[0]).get("message") or {}).get("content") or "")
        return txt.strip()
    except Exception as e:
        log.warning("Groq generate failed: %s", e)
        return ""


def gemini_generate_text(
    prompt: str,
    model: str,
    temperature: float = 0.2,
    max_output_tokens: int = 300,
) -> str:
    if _gemini_client is None:
        return ""

    model = (model or GEMINI_CLIENT_MODEL).strip()
    try:
        resp = _gemini_client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": float(temperature), "max_output_tokens": int(max_output_tokens)},
        )
        return (getattr(resp, "text", None) or "").strip()
    except Exception as e:
        msg = _safe_text(e)
        log.warning("Gemini generate failed: %s", msg[:240])
        return ""


def llm_generate_text(
    prompt: str,
    purpose: str,
    temperature: float = 0.2,
    max_output_tokens: int = 300,
) -> str:
    prompt = (prompt or "").strip()
    if not prompt:
        return ""

    out = groq_generate_text(prompt, temperature=temperature, max_output_tokens=max_output_tokens)
    if out:
        return out

    model = GEMINI_REPAIR_MODEL if purpose in ("intent", "repair", "place", "place_extract") else GEMINI_CLIENT_MODEL
    out = gemini_generate_text(prompt, model=model, temperature=temperature, max_output_tokens=max_output_tokens)
    return out or ""


def _safe_json(text: Any) -> Dict[str, Any]:
    s = _safe_text(text).strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except Exception:
            pass
    return {}


# -----------------------
# Scenarios
# -----------------------
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "family_budget": {
        "name": "Family trip (budget-sensitive)",
        "persona": "Indian parent planning a family vacation",
        "tone": "friendly, practical",
        "hidden_slots": {
            "travellers": "2 adults + 2 kids",
            "kids_ages": "7 and 11",
            "month": "April",
            "date_window": "mid-April (flexible by ~1 week)",
            "departure_city": "Delhi",
            "duration": "5–7 nights (not fixed)",
            "budget": "₹60k–₹90k per adult",
            "budget_includes": "flights + hotels ideally",
            "pace": "not too hectic",
            "hotel_level": "mid-range (3–4 star)",
            "food": "vegetarian-friendly preferred",
            "constraints": [
                "avoid too many hotel changes",
                "prefer morning flights",
                "kids get bored with long museums",
            ],
            "activity_style": "sightseeing + one light activity",
            "avoid": ["scuba", "extreme adventure sports"],
            "international_pref": "open only if visa is easy and total cost makes sense",
        },
        "max_turns": 12,
    },
    "honeymoon_premium": {
        "name": "Honeymoon (premium preferences)",
        "persona": "Newly married customer planning honeymoon",
        "tone": "excited but picky",
        "hidden_slots": {
            "travellers": "2 adults",
            "month": "March",
            "date_window": "mid-March (flexible by a few days)",
            "departure_city": "Mumbai",
            "duration": "6–8 nights (not fixed)",
            "budget": "₹2.0L–₹3.5L total",
            "budget_includes": "flights + hotels",
            "pace": "relaxed",
            "hotel_level": "premium (4–5 star)",
            "food": "no strong preference",
            "constraints": ["avoid hectic transfers"],
            "activity_style": "sightseeing + one light experience",
            "avoid": ["scuba", "extreme adventure sports"],
            "international_pref": "open, but prefers hassle-free",
        },
        "max_turns": 12,
    },
}

# -----------------------
# Regex signals (cheap path)
# -----------------------
ABUSE_RX = re.compile(r"\b(kill yourself|kys|die|suicide|idiot|stupid|bitch|fuck you)\b", re.I)

TRAVEL_HINT = re.compile(
    r"\b(trip|travel|package|flight|hotel|itinerary|day[- ]wise|plan|visa|permit|booking|tour|holiday|vacation|"
    r"honeymoon|recommend|suggest|destination|visit|go to|option|options|kids?|family|weather|season|spring|summer|"
    r"transfers?|inclusions?|stay|resort|area|north|south)\b",
    re.I,
)

ASK_PAX_RX = re.compile(
    r"\b(how many(?:\s+people|\s+travell?ers|\s+adults|\s+kids|\s+children)?|number of (?:people|travell?ers|adults|kids|children))\b",
    re.I,
)
ASK_DATES_RX = re.compile(r"\b(when|dates?|month|week|timing)\b", re.I)
ASK_DURATION_RX = re.compile(r"\b(nights?|days?|how long|duration|stay)\b", re.I)
ASK_BUDGET_RX = re.compile(r"\b(budget|price range|cost|spend|per adult|per person)\b", re.I)
ASK_FROM_RX = re.compile(r"\b(from where|from which city|where are you flying|departure city|leaving from|origin)\b", re.I)
ASK_KIDS_RX = re.compile(r"\b(kids?\s*age|children\s*age|how old are (?:the )?kids)\b", re.I)

ASK_PREF_RX = re.compile(
    r"\b(prefer|preference|pace|relaxed|hectic|comfortable|low key|adventurous|what would you prefer|which would you prefer)\b",
    re.I,
)

ASK_ACT_RX = re.compile(r"\b(what activities|activities do you like|do you like to do|scuba|hiking|water sports)\b", re.I)
ASK_DOMINT_RX = re.compile(r"\b(india or abroad|domestic or international|abroad|international)\b", re.I)

SUGGEST_RX = re.compile(r"\b(i suggest|i recommend|recommend|suggest|better option|options are|you can go to|consider|best options)\b", re.I)
MISINTERPRET_RX = re.compile(r"\b(so you (mean|want|would)|so basically|right\?)\b", re.I)
LONGHAUL_RX = re.compile(r"\b(usa|united states|europe|uk|united kingdom|london|paris|new york|los angeles|canada|australia|new zealand)\b", re.I)
CONTACT_RX = re.compile(r"\b(whatsapp|phone number|contact number|your number|share your number|text you|message you)\b", re.I)

OFFER_PLAN_RX = re.compile(r"\b(send|share)\s+(the\s+)?(plan|itinerary|details|quote)\b|\bdo you have any other questions\b", re.I)

QUESTIONISH = re.compile(r"\?|^(can you|could you|would you|do you|are you|what|which|where|when|how|tell me)\b", re.I)
MAX_Q_PER_TURN = 3


def _looks_like_question(text: str) -> bool:
    return bool(QUESTIONISH.search((text or "").strip()))


def extract_agent_intent_fallback(agent_text: str) -> Dict[str, Any]:
    t = agent_text or ""
    qlike = _looks_like_question(t)

    qs: List[Dict[str, Any]] = []
    qid_i = 1

    def add_q(qtype: str, span: str):
        nonlocal qid_i
        qs.append({"qid": f"q{qid_i}", "type": qtype, "span": span})
        qid_i += 1

    if qlike and ASK_FROM_RX.search(t):
        add_q("departure_city", "from where")
    if qlike and ASK_BUDGET_RX.search(t):
        add_q("budget", "budget")
    if qlike and ASK_PAX_RX.search(t):
        add_q("travellers", "travellers")
    if qlike and ASK_DATES_RX.search(t):
        add_q("dates", "dates")
    if qlike and ASK_DURATION_RX.search(t):
        add_q("duration", "duration")
    if qlike and ASK_KIDS_RX.search(t):
        add_q("kids_ages", "kids age")
    if qlike and ASK_PREF_RX.search(t):
        add_q("preferences", "preferences")
    if qlike and ASK_ACT_RX.search(t):
        add_q("activities", "activities")
    if qlike and ASK_DOMINT_RX.search(t):
        add_q("domestic_vs_international", "domestic/international")

    qs = qs[:MAX_Q_PER_TURN]

    signals = {
        "is_suggesting_options": bool(SUGGEST_RX.search(t)),
        "mentions_itinerary": bool(re.search(r"\bitinerary|day[- ]wise|plan\b", t, re.I)),
        "mentions_price_or_budget": bool(ASK_BUDGET_RX.search(t)),
        "mentions_domestic_international": bool(ASK_DOMINT_RX.search(t)),
        "mentions_longhaul_expensive": bool(LONGHAUL_RX.search(t)),
        "misinterpretation_check": bool(MISINTERPRET_RX.search(t)),
        "asked_weather": bool(re.search(r"\b(weather|season|hot|cold|monsoon)\b", t, re.I)),
        "contact_request": bool(CONTACT_RX.search(t)),
        "offer_plan": bool(OFFER_PLAN_RX.search(t)),
    }

    on_topic = bool(TRAVEL_HINT.search(t) or "?" in t or signals["is_suggesting_options"] or signals["contact_request"])
    return {"on_topic": on_topic, "questions": qs, "signals": signals}


def extract_agent_intent(agent_text: str) -> Dict[str, Any]:
    return extract_agent_intent_fallback(agent_text)


# -----------------------
# State / Memory helpers
# -----------------------
def build_state_for_scenario(scenario: Dict[str, Any]) -> Dict[str, Any]:
    slots = dict(scenario["hidden_slots"])
    return {
        "slots": slots,
        "disclosed": {k: False for k in slots.keys()},
        "memory": {
            "phase": "discovery",  # discovery -> options -> evaluation -> planning
            "last_client_text": "",
            "asked_weather_places": {},
            "asked_kids_places": {},
            "asked_visa_places": {},
            "selected_place": "",
            "selected_place_key": "",
            "selected_at_turn": -1,
            "last_places": [],
            "asked_best_value_once": False,
            "planning_asked": {
                "itinerary": False,
                "inclusions": False,
                "stay_areas": False,
                "transfers": False,
                "quote": False,
            },
        },
        "max_turns": int(scenario.get("max_turns", 12)),
    }


def create_session_dict(scenario_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
    if scenario_id is None:
        scenario_id = random.choice(list(SCENARIOS.keys()))
    scenario = SCENARIOS[scenario_id]
    return {
        "session_id": session_id or "",
        "turn": 0,
        "done": False,
        "scenario_id": scenario_id,
        "scenario": scenario,
        "state": build_state_for_scenario(scenario),
        "transcript": [],
        "report": None,
    }


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9₹ ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def mark_disclosed(state: Dict[str, Any], qtype: str) -> None:
    d = state["disclosed"]
    if qtype == "travellers":
        d["travellers"] = True
    elif qtype == "kids_ages":
        d["kids_ages"] = True
    elif qtype == "dates":
        d["month"] = True
        d["date_window"] = True
    elif qtype == "duration":
        d["duration"] = True
    elif qtype == "budget":
        d["budget"] = True
        d["budget_includes"] = True
    elif qtype == "departure_city":
        d["departure_city"] = True
    elif qtype == "preferences":
        d["pace"] = True
        d["constraints"] = True
        d["food"] = True
        d["hotel_level"] = True
    elif qtype == "activities":
        d["activity_style"] = True
        d["avoid"] = True
    elif qtype == "domestic_vs_international":
        d["international_pref"] = True


def budget_tier(state: Dict[str, Any]) -> int:
    b = (state["slots"].get("budget") or "").lower()
    if "₹60" in b or "₹90" in b or "60k" in b or "90k" in b:
        return 2
    if "3.5" in b or "₹2.0" in b or "2.0l" in b or "3.5l" in b:
        return 3
    return 3


def answer_for_question(state: Dict[str, Any], qtype: str) -> str:
    slots = state["slots"]
    mark_disclosed(state, qtype)

    if qtype == "departure_city":
        constraints = " ".join(slots.get("constraints", []) or []).lower()
        extra = " Prefer morning flights if possible." if "morning flights" in constraints else ""
        return f"From {slots.get('departure_city','Delhi')}.{extra}".strip()

    if qtype == "budget":
        inc = slots.get("budget_includes", "")
        if inc:
            return f"Budget is around {slots.get('budget','a mid-range budget')}, ideally including {inc}."
        return f"Budget is around {slots.get('budget','a mid-range budget')}."

    if qtype == "travellers":
        return f"We’re {slots.get('travellers','two adults')}."

    if qtype == "kids_ages":
        if "kids" in (slots.get("travellers", "")).lower():
            return f"Kids are {slots.get('kids_ages','school-age')}."
        return "No kids—just two of us."

    if qtype == "dates":
        window = slots.get("date_window", "")
        if window:
            return f"{slots.get('month','April')}—probably {window}, but we can shift a bit if it helps."
        return f"{slots.get('month','April')}, and we’re flexible."

    if qtype == "duration":
        return f"{slots.get('duration','around a week')}."

    if qtype == "preferences":
        pace = slots.get("pace", "not too hectic")
        constraints = slots.get("constraints", []) or []
        picked = []
        for x in constraints:
            lx = x.lower()
            if "hotel" in lx or "morning" in lx or "hectic" in lx:
                picked.append(x)
        if not picked:
            picked = constraints[:2]
        bits = [f"Something {pace}."]
        if picked:
            bits.append(f"And {', '.join(picked[:2])}.")
        return " ".join(bits).strip()

    if qtype == "activities":
        avoid = slots.get("avoid", []) or []
        if any("scuba" in (a or "").lower() for a in avoid):
            return "Mostly sightseeing, plus one light activity like a short hike or a park—no scuba."
        return f"{slots.get('activity_style','Mostly sightseeing')}, nothing too extreme."

    if qtype == "domestic_vs_international":
        pref = slots.get("international_pref", "open to both if it makes sense")
        return f"I’m open, but I’d prefer whatever is smoother and good value. {pref}."

    return "Got it."


def finalize_client_text(state: Dict[str, Any], parts: List[str], followup: str) -> str:
    text = " ".join([p.strip() for p in parts if p and p.strip()]).strip()
    if followup:
        if text and text[-1] not in ".!?":
            text += "."
        text = (text + " " + followup.strip()).strip()

    text = re.sub(r"\s+", " ", text).strip()

    last = state["memory"].get("last_client_text", "")
    if last and _norm(text) == _norm(last):
        text = "Got it—can you share a simple day-wise plan and what’s included?"
    state["memory"]["last_client_text"] = text
    return text


def make_llm_fn_for_places() -> Callable[[str], str]:
    def _fn(prompt: str) -> str:
        return llm_generate_text(prompt, purpose="place", temperature=0.1, max_output_tokens=220)

    return _fn


_PLACE_STOP = {
    "premium",
    "locations",
    "location",
    "place",
    "places",
    "option",
    "options",
    "destination",
    "destinations",
    "like",
    "such",
    "as",
    "a",
    "an",
    "the",
    "good",
    "best",
    "better",
    "great",
    "nice",
}


def llm_extract_place_names(agent_text: str, max_places: int = 3) -> List[str]:
    if not GROQ_API_KEY and _gemini_client is None:
        return []

    prompt = f"""
Return ONLY strict JSON: {{"places":[string]}}.

Extract travel destination names mentioned/suggested in this message.
Rules:
- Return ONLY real place names (city/region/country/island).
- Ignore generic phrases like "premium locations" or "best option".
- Up to {max_places} places.

TEXT:
{agent_text}
""".strip()

    out = llm_generate_text(prompt, purpose="place_extract", temperature=0.1, max_output_tokens=160)
    data = _safe_json(out)
    places = data.get("places", [])
    if not isinstance(places, list):
        return []

    cleaned: List[str] = []
    for p in places:
        if not isinstance(p, str):
            continue
        s = re.sub(r"[^a-zA-Z\s&\-']", " ", p).strip()
        s = re.sub(r"\s+", " ", s)
        if not s:
            continue
        toks = [w for w in s.split() if w.lower() not in _PLACE_STOP]
        s = " ".join(toks).strip()
        if s and s.lower() not in [x.lower() for x in cleaned]:
            cleaned.append(s)
        if len(cleaned) >= max_places:
            break
    return cleaned


def _accepted_fit_from_reaction(reaction: str) -> bool:
    r = (reaction or "").lower()
    if "sounds interesting" not in r:
        return False
    bad = ("pricey side", "go over budget", "worried it might")
    return not any(b in r for b in bad)


# -----------------------
# Planning helpers
# -----------------------
PLANNING_COVER_RX = {
    "itinerary": re.compile(r"\b(itinerary|day[- ]wise|day wise|plan)\b", re.I),
    "inclusions": re.compile(r"\b(inclusion|included|includes|breakfast|meals?|transfer|cab|sightseeing)\b", re.I),
    "stay_areas": re.compile(r"\b(stay|hotel|resort|area|locality|north goa|south goa)\b", re.I),
    "transfers": re.compile(r"\b(transfer|pickup|drop|airport|cab|driver|ferry)\b", re.I),
    "quote": re.compile(r"\b(quote|price|cost|budget|breakup|per person|per adult|package cost)\b", re.I),
}


def _mark_planning_coverage_from_agent(mem: Dict[str, Any], agent_text: str) -> None:
    asked = mem.get("planning_asked") or {}
    for k, rx in PLANNING_COVER_RX.items():
        if rx.search(agent_text or ""):
            asked[k] = True
    mem["planning_asked"] = asked


def _next_planning_followup(mem: Dict[str, Any], place: str) -> str:
    asked = mem.get("planning_asked") or {}
    if not asked.get("itinerary", False):
        asked["itinerary"] = True
        mem["planning_asked"] = asked
        return f"Great—can you share a simple day-wise itinerary for {place} (5–7 nights), not too hectic?"

    if not asked.get("inclusions", False):
        asked["inclusions"] = True
        mem["planning_asked"] = asked
        return "What exactly is included in the package (hotel, breakfast, transfers/cab, sightseeing)?"

    if not asked.get("stay_areas", False):
        asked["stay_areas"] = True
        mem["planning_asked"] = asked
        return f"Which areas are best to stay in for families in {place}? 2–3 mid-range hotel options would help."

    if not asked.get("transfers", False):
        asked["transfers"] = True
        mem["planning_asked"] = asked
        return "How are the transfers/local commute—any long drives, and can we keep hotel changes minimal?"

    if not asked.get("quote", False):
        asked["quote"] = True
        mem["planning_asked"] = asked
        return "Can you share an approximate total cost with a simple breakup (flights vs hotel vs local)?"

    return "Okay—what would be the next step to proceed if we want to book?"


# -----------------------
# step() (UPDATED FLOW + MEMORY FIX)
# -----------------------
def step(session: Dict[str, Any], agent_text: Optional[str]) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]], bool]:
    state = session["state"]
    mem = state["memory"]

    # OPENER
    if agent_text is None:
        session["turn"] += 1
        mem["phase"] = "discovery"
        opener = (
            "Hi! I’m looking to plan a trip "
            f"in {state['slots'].get('month','the coming month')} "
            f"with {state['slots'].get('travellers','my family')}. "
            "I’m a bit confused between places—can you shortlist 2–3 options?"
        )
        mem["last_client_text"] = opener
        analysis = {"on_topic": True, "questions": [], "signals": {"opener": True}, "answered_all": True}
        done = session["turn"] >= state.get("max_turns", 12)
        return opener, analysis, [], done

    agent_text = (agent_text or "").strip()
    session["transcript"].append({"speaker": "salesperson", "text": agent_text})

    # Mark planning coverage if agent already answered itinerary/inclusions/etc
    _mark_planning_coverage_from_agent(mem, agent_text)

    # ABUSE
    if ABUSE_RX.search(agent_text):
        client_text = "Let’s keep it professional. What details do you need for the trip?"
        mem["last_client_text"] = client_text
        analysis = {"on_topic": True, "questions": [], "signals": {"abuse": True}, "answered_all": True}
        session["turn"] += 1
        done = session["turn"] >= state.get("max_turns", 12)
        return client_text, analysis, [], done

    intent = extract_agent_intent(agent_text)
    on_topic = bool(intent.get("on_topic", True))
    questions = intent.get("questions", []) or []
    signals = intent.get("signals", {}) or {}

    # WhatsApp/phone asks
    if signals.get("contact_request", False) or CONTACT_RX.search(agent_text):
        place = mem.get("selected_place") or "the trip"
        client_text = f"I’d prefer to keep it on chat for now—can you share the itinerary + inclusions here for {place}?"
        mem["last_client_text"] = client_text
        session["turn"] += 1
        done = session["turn"] >= state.get("max_turns", 12)
        return client_text, {"on_topic": True, "questions": [], "signals": {**signals, "contact_handled": True}}, [], done

    # Detect destinations
    llm_fn = make_llm_fn_for_places() if (GROQ_API_KEY or _gemini_client is not None) else None
    places: List[str] = extract_places(agent_text, max_places=3, llm_generate=llm_fn)

    if not places and (SUGGEST_RX.search(agent_text) or re.search(r"\boption\b|\boptions\b", agent_text, re.I)):
        places = llm_extract_place_names(agent_text, max_places=3)

    if not places and mem.get("last_places"):
        if TRAVEL_HINT.search(agent_text):
            places = mem["last_places"][:1]

    if places:
        on_topic = True
        signals["is_suggesting_options"] = True
        signals["mentioned_places"] = places
        mem["last_places"] = places[:]

    # ✅ MEMORY FIX:
    # If we've already selected a place, we should not treat short confirmations
    # ("yes", "it's beautiful", etc.) as off-topic.
    if mem.get("selected_place"):
        on_topic = True

    # Off-topic fallback
    if (not on_topic) and (not questions) and (not TRAVEL_HINT.search(agent_text)) and (not places):
        client_text = "Sorry—are we discussing the trip? What details do you need from me?"
        mem["last_client_text"] = client_text
        analysis = {"on_topic": False, "questions": [], "signals": {"off_topic": True}, "answered_all": True}
        session["turn"] += 1
        done = session["turn"] >= state.get("max_turns", 12)
        return client_text, analysis, [], done

    # Phase auto-advance: evaluation -> planning after 1 agent reply (if no new places)
    phase = (mem.get("phase") or "discovery").strip().lower()
    selected_place = (mem.get("selected_place") or "").strip()
    selected_at_turn = int(mem.get("selected_at_turn", -1))

    if selected_place and phase == "evaluation":
        if (session["turn"] - selected_at_turn) >= 1 and (not places):
            mem["phase"] = "planning"
            phase = "planning"

    answers: List[Dict[str, Any]] = []
    parts: List[str] = []

    for q in questions:
        qid = q.get("qid")
        qtype = q.get("type")
        if not qid or not qtype:
            continue
        ans = answer_for_question(state, qtype)
        answers.append({"qid": qid, "type": qtype, "answer": ans})
        parts.append(ans)

    followup = ""

    # If agent offers plan and we have selected place -> planning
    if signals.get("offer_plan") and selected_place:
        mem["phase"] = "planning"
        phase = "planning"
        followup = _next_planning_followup(mem, selected_place)

    # If agent suggests more places after selection: anchor back to selected place (no loop)
    if (not followup) and places and selected_place and phase in ("planning", "evaluation"):
        followup = (
            f"{selected_place} still sounds good for us. "
            "Unless you feel it’s a bad fit, can we finalize that first—share the day-wise plan + inclusions + best stay areas?"
        )

    # Reaction if places suggested AND we are not locked into planning yet
    if (not followup) and places and (not selected_place or phase in ("discovery", "options")):
        mem["phase"] = "options"
        phase = "options"

        cards = [get_place_card(p, llm_generate=llm_fn) for p in places[:2]]
        has_kids = "kids" in (state["slots"].get("travellers") or "").lower()
        month = (state["slots"].get("month") or "").strip() or "April"

        p0 = normalize_place(places[0])
        asked_weather_places = mem.get("asked_weather_places", {}) or {}
        asked_kids_places = mem.get("asked_kids_places", {}) or {}
        asked_visa_places = mem.get("asked_visa_places", {}) or {}

        reaction = build_place_reaction(
            cards,
            has_kids=has_kids,
            budget_tier=budget_tier(state),
            already_asked_weather=bool(asked_weather_places.get(p0, False)),
            already_asked_kids=bool(asked_kids_places.get(p0, False)),
            already_asked_visa=bool(asked_visa_places.get(p0, False)),
            travel_month=month,
        )

        if reaction:
            followup = reaction

            if re.search(r"\bhow is it in\b", reaction, re.I):
                asked_weather_places[p0] = True
                mem["asked_weather_places"] = asked_weather_places
            if re.search(r"\beasy with kids\b", reaction, re.I):
                asked_kids_places[p0] = True
                mem["asked_kids_places"] = asked_kids_places
            if re.search(r"\bvisa\b|\bprocessing time\b", reaction, re.I):
                asked_visa_places[p0] = True
                mem["asked_visa_places"] = asked_visa_places

            # Accept selection (moves to evaluation)
            if _accepted_fit_from_reaction(reaction):
                mem["selected_place"] = cards[0].get("canonical") or places[0].title()
                mem["selected_place_key"] = p0
                mem["selected_at_turn"] = session["turn"]
                mem["phase"] = "evaluation"
                phase = "evaluation"

    # If planning: ALWAYS ask planning follow-ups, never "suggest options"
    if (not followup) and (mem.get("phase") == "planning") and (mem.get("selected_place") or ""):
        followup = _next_planning_followup(mem, mem["selected_place"])

    # If evaluation but no followup: ask one bridging question then planning will kick in next
    if (not followup) and (mem.get("phase") == "evaluation") and (mem.get("selected_place") or ""):
        followup = "Sounds good. Can you confirm whether it will be hectic with kids and the best flight timings?"

    # If no questions and no followup: controlled default by phase
    if not questions and not followup:
        if mem.get("selected_place"):
            mem["phase"] = "planning"
            followup = _next_planning_followup(mem, mem["selected_place"])
        else:
            mem["phase"] = "discovery"
            followup = "What 2–3 places would you suggest based on this?"

    client_text = finalize_client_text(state, parts, followup)

    session["turn"] += 1
    done = session["turn"] >= state.get("max_turns", 12)
    return client_text, {"on_topic": True, "questions": questions, "signals": signals, "answered_all": True}, answers, done


# -----------------------
# STT Repair (optional)
# -----------------------
USE_STT_REPAIR = True
STT_REPAIR_MIN_CHARS = 8

_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "so",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "at",
    "from",
    "as",
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "i",
    "you",
    "we",
    "they",
    "he",
    "she",
    "it",
    "my",
    "your",
    "our",
    "their",
    "this",
    "that",
    "these",
    "those",
    "what",
    "when",
    "where",
    "which",
    "who",
    "how",
    "why",
    "please",
    "plz",
    "kindly",
    "just",
    "okay",
    "ok",
    "yeah",
    "yep",
    "no",
    "yes",
    "do",
    "did",
    "does",
    "can",
    "could",
    "would",
    "should",
    "will",
    "me",
    "us",
    "them",
    "him",
    "her",
}


def _tok_words(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", (s or "").lower())


def accept_repair(raw: str, clean: str) -> Tuple[bool, str]:
    raw = (raw or "").strip()
    clean = (clean or "").strip()
    if not raw or not clean:
        return False, "empty"

    r = _tok_words(raw)
    c = _tok_words(clean)
    if not c:
        return False, "no_tokens"

    raw_set = set(r)
    raw_content = {w for w in raw_set if w not in _STOPWORDS}
    clean_content = {w for w in set(c) if w not in _STOPWORDS}

    new_content = clean_content - raw_content
    if new_content:
        return False, f"introduced_new_words:{sorted(list(new_content))[:6]}"

    overlap = sum(1 for w in c if w in raw_set)
    overlap_ratio = overlap / max(1, len(c))
    if overlap_ratio < 0.75:
        return False, f"low_overlap:{overlap_ratio:.2f}"

    if len(clean) > max(170, len(raw) * 2):
        return False, "too_long"

    return True, "ok"


def repair_stt_with_llm(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        return {"clean": "", "can_repair": False, "reason": "empty"}

    if not GROQ_API_KEY and _gemini_client is None:
        return {"clean": raw, "can_repair": False, "reason": "no_llm_configured"}

    prompt = f"""
Return ONLY strict JSON:
{{"clean": string, "can_repair": boolean, "reason": string}}

Task: Convert messy speech-to-text into a clean single utterance said by a travel agent.

Rules:
- Keep SAME meaning. Do NOT add new facts/prices/destinations/dates/policies.
- Remove repetition/fillers, fix grammar, natural spoken English (India).
- clean: 1 sentence or 2 short sentences, max ~170 chars.

RAW_STT:
{raw}
""".strip()

    out = llm_generate_text(prompt, purpose="repair", temperature=0.2, max_output_tokens=220)
    data = _safe_json(out)
    clean = (data.get("clean") or "").strip()
    can_repair = bool(data.get("can_repair", True))
    reason = (data.get("reason") or "").strip()

    if not clean:
        return {"clean": raw, "can_repair": False, "reason": reason or "no_clean_text"}

    clean = re.sub(r"\s+", " ", clean).strip()
    if len(clean) > max(220, len(raw) * 3):
        return {"clean": raw, "can_repair": False, "reason": "too_long_after_repair"}

    return {"clean": clean, "can_repair": can_repair, "reason": reason or "ok"}


# -----------------------
# API Models
# -----------------------
class StartReq(BaseModel):
    session_id: str = Field(...)


class TurnReq(BaseModel):
    session_id: str = Field(...)
    user_text: str = Field("", description="Agent text (typed or STT)")


# -----------------------
# Pages + API
# -----------------------
def get_or_load_session(session_id: str) -> Optional[Dict[str, Any]]:
    session = SESSIONS.get(session_id)
    if session:
        return session

    raw = load_session_from_disk(session_id)
    if not raw:
        return None

    scenario_id = raw.get("scenario_id")
    if scenario_id not in SCENARIOS:
        return None

    session = create_session_dict(scenario_id=scenario_id, session_id=session_id)
    session["turn"] = int(raw.get("turn", 0))
    session["done"] = bool(raw.get("done", False))
    session["state"] = raw.get("state") or session["state"]
    session["transcript"] = raw.get("transcript") or []
    session["report"] = raw.get("report", None)
    SESSIONS[session_id] = session
    return session


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    session_id = str(uuid.uuid4())
    session = create_session_dict(session_id=session_id)
    session["session_id"] = session_id
    SESSIONS[session_id] = session
    save_session(session)
    return templates.TemplateResponse("index.html", {"request": request, "session_id": session_id})


@app.get("/health")
@limiter.limit("60/minute")
async def health(request: Request):
    return {
        "ok": True,
        "providers": {
            "groq_enabled": bool(GROQ_API_KEY),
            "groq_model": GROQ_MODEL,
            "gemini_enabled": _gemini_client is not None,
            "gemini_client_model": GEMINI_CLIENT_MODEL,
            "gemini_repair_model": GEMINI_REPAIR_MODEL,
        },
        "twilio_router_loaded": bool(twilio_router is not None),
    }


@app.post("/api/start")
@limiter.limit("10/minute")
async def api_start(request: Request, payload: StartReq):
    session = get_or_load_session(payload.session_id)
    if not session:
        return JSONResponse({"error": "Invalid session"}, status_code=400)

    client_text, analysis, answers, done = step(session, None)
    session["transcript"].append({"speaker": "client", "text": client_text})
    session["done"] = done
    save_session(session)
    return {"client_text": client_text, "analysis": analysis, "answers": answers, "done": done}


@app.post("/api/start/")
@limiter.limit("10/minute")
async def api_start_slash(request: Request, payload: StartReq):
    return await api_start(request, payload)


@app.post("/api/turn")
@limiter.limit("30/minute")
async def api_turn(request: Request, payload: TurnReq):
    session = get_or_load_session(payload.session_id)
    if not session:
        return JSONResponse({"error": "Invalid session"}, status_code=400)

    user_text_raw = (payload.user_text or "").strip()

    if session.get("done"):
        return {
            "client_text": "Session already ended.",
            "done": True,
            "user_text_raw": user_text_raw,
            "user_text_clean": user_text_raw,
            "stt_repaired": False,
            "stt_reason": "done",
        }

    user_text_clean = user_text_raw
    stt_repaired = False
    stt_reason = ""

    if USE_STT_REPAIR and len(user_text_raw) >= STT_REPAIR_MIN_CHARS:
        try:
            repaired = repair_stt_with_llm(user_text_raw)
            candidate = (repaired.get("clean") or "").strip()
            if repaired.get("can_repair", False) and candidate:
                ok, why = accept_repair(user_text_raw, candidate)
                if ok:
                    user_text_clean = candidate
                    stt_repaired = (user_text_clean != user_text_raw)
                    stt_reason = repaired.get("reason", "") or why
                else:
                    stt_reason = why
            else:
                stt_reason = repaired.get("reason", "cannot_repair")
        except Exception as e:
            stt_reason = f"repair_error: {e}"

    client_text, analysis, answers, done = step(session, user_text_clean)
    session["transcript"].append({"speaker": "client", "text": client_text})
    session["done"] = done
    save_session(session)

    return {
        "client_text": client_text,
        "analysis": analysis,
        "answers": answers,
        "done": done,
        "user_text_raw": user_text_raw,
        "user_text_clean": user_text_clean,
        "stt_repaired": stt_repaired,
        "stt_reason": stt_reason,
    }


@app.post("/api/turn/")
@limiter.limit("30/minute")
async def api_turn_slash(request: Request, payload: TurnReq):
    return await api_turn(request, payload)


# -----------------------
# REPORTING
# -----------------------
def _count_words(s: str) -> int:
    return len(re.findall(r"\b[\w']+\b", s or ""))


def build_performance_report(session: Dict[str, Any]) -> Dict[str, Any]:
    transcript = session.get("transcript", []) or []
    sales_lines = [t.get("text", "") for t in transcript if t.get("speaker") == "salesperson"]
    client_lines = [t.get("text", "") for t in transcript if t.get("speaker") == "client"]

    sales_words = sum(_count_words(x) for x in sales_lines)
    client_words = sum(_count_words(x) for x in client_lines)
    total_words = max(1, sales_words + client_words)

    question_count = sum(1 for x in sales_lines if "?" in x)
    open_q = 0
    open_starts = ("what", "why", "how", "where", "when", "which", "tell me", "could you share")
    for x in sales_lines:
        t = (x or "").strip().lower()
        if "?" in t and t.startswith(open_starts):
            open_q += 1

    salesperson_share = round((sales_words / total_words) * 100)

    def has_any(rx: str) -> bool:
        r = re.compile(rx, re.I)
        return any(r.search(x or "") for x in sales_lines)

    checks_discovery = [
        {"item": "Asked departure city / origin", "met": has_any(r"\bfrom where\b|\bwhere are you flying\b|\bdeparture city\b")},
        {"item": "Asked dates/month", "met": has_any(r"\bwhen\b|\bdates?\b|\bmonth\b")},
        {"item": "Asked duration (nights/days)", "met": has_any(r"\bnights?\b|\bdays?\b|\bduration\b")},
        {"item": "Asked budget", "met": has_any(r"\bbudget\b|\bcost\b|\bprice\b")},
        {"item": "Asked pace/preferences", "met": has_any(r"\brelaxed\b|\bhectic\b|\bprefer\b|\bpreference\b")},
    ]

    checks_structure = [
        {"item": "Suggested destinations/options", "met": has_any(r"\brecommend\b|\bsuggest\b|\boptions\b|\bconsider\b|\byou can go\b|\bbetter option\b")},
        {"item": "Kept it professional", "met": not any(ABUSE_RX.search(x or "") for x in sales_lines)},
    ]

    def score_cat(name: str, weight: int, checks: list) -> Dict[str, Any]:
        met = sum(1 for c in checks if c["met"])
        total = max(1, len(checks))
        points = round((met / total) * weight)
        return {"name": name, "weight": weight, "points": points, "checks": checks}

    categories = [
        score_cat("Discovery", 45, checks_discovery),
        score_cat("Structure", 35, checks_structure),
        score_cat("Professionalism", 20, [{"item": "No abusive language", "met": not any(ABUSE_RX.search(x or "") for x in sales_lines)}]),
    ]

    overall = sum(c["points"] for c in categories)

    evidence = []
    for x in sales_lines:
        if "?" in x or re.search(r"\brecommend\b|\bsuggest\b|\boptions\b|\bbetter option\b", x, re.I):
            evidence.append(x.strip())
        if len(evidence) >= 8:
            break

    strengths = []
    improvements = []

    if sum(c["met"] for c in checks_discovery) >= 3:
        strengths.append("Good discovery coverage (origin/budget/duration/preferences).")
    if question_count >= 3:
        strengths.append("You kept the conversation interactive with questions.")

    missing = [c["item"] for c in checks_discovery if not c["met"]]
    if missing:
        improvements.append("Improve discovery: " + ", ".join(missing[:3]) + ".")

    if salesperson_share > 70:
        improvements.append("Reduce talk-time: ask 1–2 questions, then pause to listen.")

    if open_q == 0 and question_count > 0:
        improvements.append("Use more open-ended questions (What/Which/How) to understand needs.")

    if not has_any(r"\bitinerary\b|\bday[- ]wise\b|\bplan\b"):
        improvements.append("After suggesting places, offer a rough day-wise plan + 2 hotel tiers.")

    if not improvements:
        improvements.append("Do another run and focus on discovery → options → close.")

    return {
        "overall_score": int(overall),
        "metrics": {
            "salesperson_word_share_percent": int(salesperson_share),
            "question_count": int(question_count),
            "open_question_count": int(open_q),
        },
        "categories": categories,
        "evidence": evidence,
        "strengths": strengths[:6],
        "improvements": improvements[:8],
    }


@app.post("/api/end")
@limiter.limit("10/minute")
async def api_end(request: Request, payload: Dict[str, Any]):
    session_id = (payload.get("session_id") or "").strip()
    session = get_or_load_session(session_id)
    if not session:
        return JSONResponse({"error": "Invalid session"}, status_code=400)

    session["done"] = True
    report = build_performance_report(session)
    session["report"] = report
    save_session(session)

    return {"ok": True, "report_url": f"/report/{session_id}"}


@app.post("/api/end/")
@limiter.limit("10/minute")
async def api_end_slash(request: Request, payload: Dict[str, Any]):
    return await api_end(request, payload)


@app.get("/report/{session_id}", response_class=HTMLResponse)
async def report_page(request: Request, session_id: str):
    session = get_or_load_session(session_id)
    if not session:
        return HTMLResponse("<h2>Invalid session</h2>", status_code=404)

    report = session.get("report") or build_performance_report(session)
    session["report"] = report
    save_session(session)
    transcript = session.get("transcript", []) or []

    return templates.TemplateResponse(
        "report.html",
        {"request": request, "session_id": session_id, "report": report, "transcript": transcript},
    )
