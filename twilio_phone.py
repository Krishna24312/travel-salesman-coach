# twilio_phone.py
from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import Response

from twilio.twiml.voice_response import VoiceResponse, Gather

router = APIRouter()

# Tuning (simple + works well for India callers too)
VOICE_LANG = os.getenv("TWILIO_VOICE_LANG", "en-IN")
VOICE_NAME = os.getenv("TWILIO_VOICE_NAME", "alice")  # built-in Twilio voice


def _twiml(xml: str) -> Response:
    return Response(content=xml, media_type="text/xml; charset=utf-8")


def _gather(prompt: Optional[str] = None) -> Gather:
    """
    Collect one spoken turn from the agent.
    Twilio will POST SpeechResult to /twilio/turn.
    """
    g = Gather(
        input="speech",
        action="/twilio/turn",
        method="POST",
        language=VOICE_LANG,
        speech_timeout="auto",
    )
    if prompt:
        g.say(prompt, voice=VOICE_NAME, language=VOICE_LANG)
    return g


@router.post("/twilio/voice")
async def twilio_voice(request: Request):
    """
    Entry point when someone calls the Twilio number.
    Uses CallSid as session_id so your report is /report/{CallSid}.
    """
    form = await request.form()
    call_sid = (form.get("CallSid") or "").strip()

    import main  # runtime import to avoid circular imports

    # Load or create a session keyed by CallSid
    session = main.get_or_load_session(call_sid)
    if not session:
        session = main.create_session_dict(session_id=call_sid)
        session["session_id"] = call_sid
        main.SESSIONS[call_sid] = session
        main.save_session(session)

    # Let the client open the scenario (same as /api/start)
    client_text, _analysis, _answers, done = main.step(session, None)
    session["transcript"].append({"speaker": "client", "text": client_text})
    session["done"] = bool(done)
    main.save_session(session)

    vr = VoiceResponse()
    vr.say(client_text, voice=VOICE_NAME, language=VOICE_LANG)

    if session["done"]:
        vr.say("Thanks. This session is complete. Goodbye.", voice=VOICE_NAME, language=VOICE_LANG)
        vr.hangup()
        return _twiml(str(vr))

    # Listen for agent reply
    vr.append(_gather())
    # If silence, re-enter same flow
    vr.redirect("/twilio/voice", method="POST")
    return _twiml(str(vr))


@router.post("/twilio/turn")
async def twilio_turn(request: Request):
    """
    Called after the agent speaks. Twilio provides SpeechResult text.
    We feed it into your existing step() and speak the client's reply.
    """
    form = await request.form()
    call_sid = (form.get("CallSid") or "").strip()
    speech = (form.get("SpeechResult") or "").strip()

    import main  # runtime import to avoid circular imports

    session = main.get_or_load_session(call_sid)
    if not session:
        session = main.create_session_dict(session_id=call_sid)
        session["session_id"] = call_sid
        main.SESSIONS[call_sid] = session
        main.save_session(session)

    if session.get("done"):
        vr = VoiceResponse()
        vr.say("This session has already ended. Goodbye.", voice=VOICE_NAME, language=VOICE_LANG)
        vr.hangup()
        return _twiml(str(vr))

    # Treat caller speech as salesperson text (optionally run your STT repair)
    user_text_clean = speech
    if getattr(main, "USE_STT_REPAIR", False) and len(speech) >= getattr(main, "STT_REPAIR_MIN_CHARS", 8):
        try:
            repaired = main.repair_stt_with_llm(speech)
            candidate = (repaired.get("clean") or "").strip()
            if repaired.get("can_repair", False) and candidate:
                ok, _why = main.accept_repair(speech, candidate)
                if ok:
                    user_text_clean = candidate
        except Exception:
            pass

    # Record salesperson line
    session["transcript"].append({"speaker": "salesperson", "text": user_text_clean})

    client_text, _analysis, _answers, done = main.step(session, user_text_clean)
    session["transcript"].append({"speaker": "client", "text": client_text})
    session["done"] = bool(done)
    main.save_session(session)

    vr = VoiceResponse()
    vr.say(client_text, voice=VOICE_NAME, language=VOICE_LANG)

    if session["done"]:
        vr.say("Ending the call now. Thanks!", voice=VOICE_NAME, language=VOICE_LANG)
        vr.hangup()
        return _twiml(str(vr))

    vr.append(_gather())
    vr.redirect("/twilio/turn", method="POST")
    return _twiml(str(vr))
