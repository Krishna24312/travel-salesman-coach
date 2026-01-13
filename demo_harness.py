# demo_harness.py
import os
import time

# Option A: run WITHOUT server (fastest)
from main import create_session_dict, scenario_next

# Option B (optional): run with server endpoints
# Set DEMO_HTTP=1 to use HTTP calls
DEMO_HTTP = os.getenv("DEMO_HTTP", "0") == "1"


def run_local_simulation(scenario_id=None):
    session = create_session_dict(scenario_id=scenario_id)
    print("=" * 80)
    print(f"LOCAL DEMO | scenario={session['scenario_id']}")
    print("=" * 80)

    # Start turn
    client_line, done = scenario_next(session, None)
    session["transcript"].append({"speaker": "client", "text": client_line})
    print("CLIENT:", client_line)

    tests = [
        # multi-question
        "Hi—can you tell me how many people are travelling, for how many nights, and which month?",
        # user answers travellers + nights but not month: customer should only answer what asked of them? (here customer is responding)
        "Okay, and what is your budget range?",
        # activities list (the big bug you complained about)
        "Okay. So would you prefer hiking, scuba diving, water sports, adventure parks, water parks? What would you prefer?",
        # repeated activities list (should not repeat the SAME generic line)
        "So hiking, scuba diving, water sports, adventure parks, or water parks—what do you prefer?",
        # non-question agent line
        "Alright, I will check and get back to you.",
        # off-topic
        "By the way do you watch football",
        # abuse
        "you are stupid, fuck you",
        # inclusions/policy
        "What all is included and what's the cancellation policy?",
        # next step
        "Cool, should we book now? I can send payment link.",
    ]

    for i, agent in enumerate(tests, 1):
        print("\n" + "-" * 80)
        print(f"AGENT {i}:", agent)
        client_line, done = scenario_next(session, agent)
        session["transcript"].append({"speaker": "client", "text": client_line})
        print("CLIENT:", client_line)
        if done:
            break

    print("\n" + "=" * 80)
    print("DONE. Transcript length:", len(session["transcript"]))
    print("=" * 80)


def run_http_simulation(base_url="http://127.0.0.1:8000"):
    import requests

    print("=" * 80)
    print("HTTP DEMO (requires server running)")
    print("=" * 80)

    # Create browser session by loading home? In your UI flow you already create session_id in HTML.
    # For harness, we fake it by calling "/" to get a new session_id is hard unless you parse HTML.
    # So easiest: open your web UI, copy session_id from DOM, paste here.
    session_id = os.getenv("SESSION_ID", "").strip()
    if not session_id:
        print("Set SESSION_ID env var first (from your web UI). Example:")
        print("  SESSION_ID=... DEMO_HTTP=1 python demo_harness.py")
        return

    r = requests.post(f"{base_url}/api/start", json={"session_id": session_id}, timeout=20)
    print("CLIENT:", r.json().get("client_text"))

    tests = [
        "Okay. So would you prefer hiking, scuba diving, water sports, adventure parks, water parks? What would you prefer?",
        "So hiking, scuba diving, water sports, adventure parks, or water parks—what do you prefer?",
        "Alright, I will check and get back to you.",
        "By the way do you watch football",
        "What all is included and what's the cancellation policy?",
    ]

    for t in tests:
        time.sleep(0.2)
        r = requests.post(f"{base_url}/api/turn", json={"session_id": session_id, "user_text": t}, timeout=20)
        js = r.json()
        print("\nAGENT:", t)
        print("CLIENT:", js.get("client_text"))
        print("pending:", js.get("pending_questions"))


if __name__ == "__main__":
    if DEMO_HTTP:
        run_http_simulation()
    else:
        # Try both scenarios back-to-back
        run_local_simulation("family_budget")
        run_local_simulation("honeymoon_premium")
