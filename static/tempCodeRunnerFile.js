const sessionId = window.__SESSION_ID__;
const chat = document.getElementById("chat");
const statusEl = document.getElementById("status");

const btnStart = document.getElementById("btnStart");
const btnTalk = document.getElementById("btnTalk");
const btnEnd = document.getElementById("btnEnd");

function addBubble(who, text) {
  const div = document.createElement("div");
  div.className = `bubble ${who}`;
  div.innerText = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function setStatus(s) {
  statusEl.innerText = `Status: ${s}`;
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  return await res.json();
}

// Browser TTS (free)
function speak(text) {
  return new Promise((resolve) => {
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.0;
    u.pitch = 1.0;
    u.onend = resolve;
    speechSynthesis.speak(u);
  });
}

// Browser STT (free) — Chrome
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null;

function setupRecognition() {
  if (!SpeechRecognition) {
    alert("SpeechRecognition not supported in this browser. Use Chrome.");
    return false;
  }
  rec = new SpeechRecognition();
  rec.lang = "en-IN";
  rec.interimResults = false;
  rec.continuous = false;
  return true;
}

let running = false;

async function handleClientReply(clientText, done) {
  addBubble("client", clientText);
  await speak(clientText);

  if (done) {
    setStatus("scenario finished — generate report");
    btnTalk.disabled = true;
    btnEnd.disabled = false;
  } else {
    setStatus("your turn — push to talk");
    btnTalk.disabled = false;
  }
}

btnStart.addEventListener("click", async () => {
  setStatus("starting...");
  btnStart.disabled = true;

  const data = await postJSON("/api/start", { session_id: sessionId });
  if (data.error) {
    alert(data.error);
    return;
  }

  btnEnd.disabled = false;
  await handleClientReply(data.client_text, data.done);
});

btnTalk.addEventListener("mousedown", async () => {
  if (!rec && !setupRecognition()) return;
  if (running) return;

  running = true;
  btnTalk.disabled = true;
  setStatus("listening...");

  rec.onresult = async (event) => {
    const text = event.results?.[0]?.[0]?.transcript || "";
    addBubble("sales", text);
    setStatus("thinking...");

    const data = await postJSON("/api/turn", { session_id: sessionId, user_text: text });
    if (data.error) {
      alert(data.error);
      running = false;
      return;
    }

    await handleClientReply(data.client_text, data.done);
    running = false;
  };

  rec.onerror = (e) => {
    console.error(e);
    setStatus("speech error — try again");
    btnTalk.disabled = false;
    running = false;
  };

  rec.onend = () => {
    // if user released quickly, onend may fire without result
    if (running) setStatus("processing...");
  };

  rec.start();
});

btnTalk.addEventListener("mouseup", () => {
  try { rec && rec.stop(); } catch (_) {}
});

btnEnd.addEventListener("click", async () => {
  setStatus("generating report...");
  const data = await postJSON("/api/end", { session_id: sessionId });
  if (data.error) {
    alert(data.error);
    return;
  }
  window.location.href = data.report_url;
});
