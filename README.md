# Travel Sales Voice Coach

A small web app I built to practice travel sales calls through roleplay. You speak as the salesperson (push-to-talk or typed) and the app replies as a realistic client persona. At the end, it generates a quick report with what went well and what to improve.


---

## What it does
- Voice/typed roleplay loop (salesperson ↔ client)
- Speech-to-text using Deepgram
- Client replies generated using Groq (Gemini optional fallback)
- Multiple scenarios (ex: budget family trip, premium honeymoon)
- End-of-call report: talk-time, question quality, discovery coverage, etc.
- FastAPI backend + simple HTML/CSS/JS UI

---

## Tech
FastAPI, Uvicorn, Jinja2 templates, vanilla JS/CSS, Deepgram STT, Groq LLM (Gemini optional)

---

## Repo layout
```txt
.
├── main.py
├── locations.py
├── requirements.txt
├── templates/
│   ├── index.html
│   └── report.html
└── static/
    ├── styles.css
    ├── app.js
    └── (images/backgrounds)
Run locally
1) Install
bash
Copy code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
2) Environment variables
Create a .env file in the project root:

env
Copy code
DEEPGRAM_API_KEY=...
GROQ_API_KEY=...

# optional
DEEPGRAM_MODEL=nova-3
DEEPGRAM_LANG=en-IN

GROQ_MODEL=llama-3.3-8b-instant
GROQ_BASE_URL=https://api.groq.com/openai/v1

# optional fallback
GEMINI_API_KEY=...
GEMINI_CLIENT_MODEL=gemini-2.5-flash
GEMINI_REPAIR_MODEL=gemini-2.5-flash-lite

LOG_LEVEL=INFO
DATA_DIR=.data
3) Start the server
bash
Copy code
uvicorn main:app --reload --port 8000
Open: http://127.0.0.1:8000

Deploy on Render
Render settings
Build Command

bash
Copy code
pip install -r requirements.txt
Start Command

bash
Copy code
uvicorn main:app --host 0.0.0.0 --port $PORT
Render environment variables
Add these in Render → Environment:

DEEPGRAM_API_KEY

GROQ_API_KEY

(optional) DEEPGRAM_MODEL, DEEPGRAM_LANG, GROQ_MODEL, GROQ_BASE_URL

(optional) GEMINI_API_KEY, GEMINI_CLIENT_MODEL, GEMINI_REPAIR_MODEL

Do not commit .env to GitHub.