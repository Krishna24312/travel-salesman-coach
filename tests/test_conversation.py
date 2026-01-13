import re
import os
import main

# -----------------------------
# Regex checks (simple + fast)
# -----------------------------
BUSY_RX = re.compile(
    r"\b(busy|short on time|limited time|keep it brief|keep it quick|running out of time)\b",
    re.I,
)
STAGE_RX = re.compile(r"(\([^)]*\))|(\[[^\]]*\])|(\*[^*]+\*)")
SENTENCE_SPLIT_RX = re.compile(r"[.!?]+")


def sentence_count(text: str) -> int:
    parts = [p.strip() for p in SENTENCE_SPLIT_RX.split(text or "") if p.strip()]
    return len(parts)


def leaked_any_slot(reply: str, slots: dict, disclosed: dict) -> bool:
    """
    Returns True if reply contains any undisclosed slot values (string match).
    Works best because your hidden_slots are fixed strings (e.g., "Delhi", "April").
    """
    r = (reply or "").lower()
    for k, v in slots.items():
        if not v or not isinstance(v, str):
            continue
        if disclosed.get(k, False):
            continue
        if v.lower() in r:
            return True
    return False


# -----------------------------
# Session factory (no FastAPI)
# -----------------------------
def make_session_for_scenario(scenario_id: str = "family_budget") -> dict:
    scenario = main.SCENARIOS[scenario_id]

    slots = dict(scenario["hidden_slots"])
    slots.setdefault("dates", None)
    slots.setdefault("destination", None)
    slots.setdefault("hotel_pref", None)

    disclosed = {k: False for k in slots.keys()}

    return {
        "turn": 0,
        "transcript": [],
        "done": False,
        "report": None,
        "scenario_id": scenario_id,
        "scenario": scenario,
        "state": {
            "max_turns": scenario.get("max_turns", 7),
            "slots": slots,
            "disclosed": disclosed,
            "progress": {
                "agent_asked_budget": False,
                "agent_asked_dates": False,
                "agent_asked_pax": False,
                "agent_asked_prefs": False,
                "agent_proposed_next_step": False,
                "agent_mentioned_inclusions": False,
                "agent_gave_price_estimate": False,
                "agent_price_estimate_text": None,
            },
            "last_agent_style": {},
            "mood": "neutral",
            "price_objection_used": False,
            "trust_objection_used": False,
            "busy_cooldown": 0,
        },
    }


def _maybe_print_transcript(session: dict, label: str = "") -> None:
    """
    To print full transcripts:
      PRINT_TRANSCRIPT=1 python -m pytest -q -s
    """
    if os.getenv("PRINT_TRANSCRIPT", "").strip() not in ("1", "true", "TRUE", "yes", "YES"):
        return

    header = f"\n\n===== TRANSCRIPT {label} ====="
    print(header)
    for x in session["transcript"]:
        print(f"{x['speaker']}: {x['text']}")
    print("===== END TRANSCRIPT =====\n")


# -----------------------------
# Core runner + assertions
# -----------------------------
def run_scripted_call(scenario_id: str, agent_turns: list, max_busy_hits: int = 2) -> dict:
    """
    Runs a full scripted call:
    - client starts (scenario_next(session, None))
    - then feed agent_turns one-by-one
    - validate each client reply
    """
    session = make_session_for_scenario(scenario_id)
    state = session["state"]
    slots = state["slots"]

    busy_hits = 0

    # ---- Client starts (no agent input) ----
    reply, done = main.scenario_next(session, None)
    session["transcript"].append({"speaker": "client", "text": reply})

    # Basic checks
    assert not STAGE_RX.search(reply), f"Stage directions leaked (start): {reply}"
    assert sentence_count(reply) <= 2, f"Too long (start): {reply}"
    assert not leaked_any_slot(reply, slots, state["disclosed"]), f"Leaked undisclosed info (start): {reply}"
    if BUSY_RX.search(reply):
        busy_hits += 1

    # ---- Feed agent turns ----
    for agent_text in agent_turns:
        reply, done = main.scenario_next(session, agent_text)
        session["transcript"].append({"speaker": "client", "text": reply})

        assert not STAGE_RX.search(reply), f"Stage directions leaked: {reply}"
        assert sentence_count(reply) <= 3, f"Reply too long: {reply}"
        assert not leaked_any_slot(reply, slots, state["disclosed"]), f"Leaked undisclosed info: {reply}"

        if BUSY_RX.search(reply):
            busy_hits += 1

    assert busy_hits <= max_busy_hits, (
        f"Busy spam detected (busy_hits={busy_hits}).\n"
        f"Transcript:\n" + "\n".join([f"{x['speaker']}: {x['text']}" for x in session["transcript"]])
    )

    _maybe_print_transcript(session, label=f"(scenario={scenario_id})")
    return session


# ==========================================================
# TESTS (edit agent_turns later as you want)
# ==========================================================

def test_family_budget_basic_discovery_flow():
    agent_turns = [
        "Hi, can you suggest a destination for a trip?",
        "Sure—how many travellers are going, any kids?",
        "When are you planning to travel? Any fixed dates or month?",
        "What budget range are you comfortable with?",
        "Any preferences—relaxed or packed itinerary, hotel type, kid-friendly activities?",
        "Based on that, a rough ballpark could be ₹70k–₹90k per adult. I can share 2 options.",
        "If you like one, I can send the proposal on WhatsApp. What number should I send it to?",
    ]
    run_scripted_call("family_budget", agent_turns)


def test_honeymoon_basic_discovery_flow():
    agent_turns = [
        "Hi, I can help. Where are you thinking of going?",
        "When are you planning to travel?",
        "What’s your total budget for the trip?",
        "Do you want a relaxed honeymoon or sightseeing-heavy?",
        "For a premium stay, a good range could be around ₹2–3L depending on destination and flights.",
        "I can share 2-3 hotel options and an itinerary—shall I send it on WhatsApp?",
    ]
    run_scripted_call("honeymoon_premium", agent_turns)


def test_off_topic_agent_gets_steered_back():
    # Agent goes off-topic → client should steer back (no stage directions, short)
    agent_turns = [
        "By the way, what do you think of crypto these days?",
        "Okay sorry—so where do you want to travel?",
        "And what’s your budget range?",
    ]
    run_scripted_call("family_budget", agent_turns)


def test_price_objection_happens_only_after_estimate():
    agent_turns = [
        "Hi, where do you want to travel?",
        "When are you planning to travel?",
        "What budget range are you comfortable with?",
        "Any preferences on hotel quality or pace?",
        # estimate appears here
        "Okay, a rough estimate could be ₹80k–₹95k per adult for a 6-night trip including stays and transfers.",
        # after estimate, client may raise price objection
        "This is standard pricing for the season; we handle everything end-to-end.",
    ]
    run_scripted_call("family_budget", agent_turns)


def test_trust_objection_triggered_by_booking_push():
    agent_turns = [
        "Hi, can you tell me the budget and dates?",
        "Great—based on that, I can offer a package around ₹2.5L total.",
        "Perfect, I can book it today. Should I send you the payment link now?",
    ]
    run_scripted_call("honeymoon_premium", agent_turns)
