// app.js (Deepgram STT via backend /api/stt)
const sessionId = window.__SESSION_ID__;
const chat = document.getElementById("chat");
const statusEl = document.getElementById("status");

const btnStart = document.getElementById("btnStart");
const btnTalk = document.getElementById("btnTalk");
const btnEnd = document.getElementById("btnEnd");

const typedInput = document.getElementById("typedInput");
const btnSend = document.getElementById("btnSend");

// Keep push-to-talk manual (more realistic + avoids accidental cutoffs)
const AUTO_LISTEN_AFTER_CLIENT = false;

// --- STT tuning knobs ---
const STT = {
  minCharsToSend: 2,
  minWordsToSend: 1,
  maxHoldMs: 20_000,
};

// ----- UI helpers -----
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
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  let data = {};
  try {
    data = await res.json();
  } catch (_) {
    data = {};
  }

  if (!res.ok) {
    const msg = data?.error || data?.detail || `HTTP ${res.status} on ${url}`;
    return { error: msg, _raw: data };
  }

  return data;
}

// ----- TTS -----
let speaking = false;
let preferredVoice = null;

function stopSpeaking() {
  try { speechSynthesis.cancel(); } catch (_) {}
  speaking = false;
}

function pickVoice() {
  const voices = speechSynthesis.getVoices();
  if (!voices || voices.length === 0) return null;

  const preferredNames = [
    "Google UK English Female",
    "Google UK English Male",
    "Google US English",
  ];
  for (const name of preferredNames) {
    const v = voices.find(vo => vo.name === name);
    if (v) return v;
  }
  const en = voices.filter(v => (v.lang || "").toLowerCase().startsWith("en"));
  if (en.length) return en[0];
  return voices[0];
}

speechSynthesis.onvoiceschanged = () => {
  preferredVoice = pickVoice();
};

function chunkText(text, maxLen = 160) {
  const chunks = [];
  let t = (text || "").trim();
  while (t.length > maxLen) {
    let cut = t.lastIndexOf(".", maxLen);
    if (cut < 60) cut = t.lastIndexOf(",", maxLen);
    if (cut < 60) cut = t.lastIndexOf(" ", maxLen);
    if (cut < 0) cut = maxLen;
    chunks.push(t.slice(0, cut + 1).trim());
    t = t.slice(cut + 1).trim();
  }
  if (t) chunks.push(t);
  return chunks;
}

function speak(text) {
  return new Promise((resolve) => {
    if (!text) return resolve();
    speechSynthesis.cancel();
    speaking = true;

    const parts = chunkText(text);
    let i = 0;

    const speakNext = () => {
      if (i >= parts.length) {
        speaking = false;
        return resolve();
      }
      const u = new SpeechSynthesisUtterance(parts[i]);
      u.rate = 1.18;
      u.pitch = 1.02;
      u.volume = 1.0;
      u.voice = preferredVoice || pickVoice();
      u.onend = () => { i++; speakNext(); };
      u.onerror = () => { speaking = false; resolve(); };
      speechSynthesis.speak(u);
    };

    speakNext();
  });
}

// ----- Deepgram STT (record audio + send to backend) -----
let callActive = false;

let mediaStream = null;
let recorder = null;
let chunks = [];
let recording = false;

let holdTimeout = null;
let lastSentNormalized = ""; // dedupe exact repeats

function normalizeForDedupe(s) {
  return (s || "")
    .toLowerCase()
    .replace(/[^\w\s]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

async function ensureMic() {
  if (mediaStream) return mediaStream;
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    }
  });
  return mediaStream;
}

function pickMimeType() {
  const preferred = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/ogg",
  ];
  for (const mt of preferred) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(mt)) return mt;
  }
  return "";
}

async function startRecording() {
  await ensureMic();

  // barge-in
  if (speaking) stopSpeaking();

  chunks = [];
  const mimeType = pickMimeType();
  recorder = new MediaRecorder(mediaStream, mimeType ? { mimeType } : undefined);

  recorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) chunks.push(e.data);
  };

  recorder.start(200);
}

function stopRecordingAndGetBlob() {
  return new Promise((resolve, reject) => {
    if (!recorder) return reject(new Error("No recorder"));
    if (recorder.state === "inactive") {
      const mime = recorder.mimeType || "audio/webm";
      return resolve(new Blob(chunks, { type: mime }));
    }

    recorder.onstop = () => {
      const mime = recorder.mimeType || "audio/webm";
      resolve(new Blob(chunks, { type: mime }));
    };

    try {
      recorder.stop();
    } catch (e) {
      reject(e);
    }
  });
}

async function transcribeWithBackend(blob) {
  const fd = new FormData();
  fd.append("audio", blob, "ptt.webm");

  const res = await fetch("/api/stt", { method: "POST", body: fd });
  let data = {};
  try { data = await res.json(); } catch (_) { data = {}; }

  if (!res.ok) {
    return { error: data?.error || data?.detail || `HTTP ${res.status} on /api/stt` };
  }
  return data;
}

async function sendTranscriptToTurn(transcript) {
  const text = (transcript || "").trim();
  const wordCount = text ? text.split(/\s+/).filter(Boolean).length : 0;

  if (!text || text.length < STT.minCharsToSend || wordCount < STT.minWordsToSend) {
    setStatus("didn't catch that — try again");
    recording = false;
    return;
  }

  const norm = normalizeForDedupe(text);
  if (norm && norm === lastSentNormalized) {
    setStatus("heard the same thing — try again");
    recording = false;
    return;
  }
  lastSentNormalized = norm;

  setStatus("thinking...");

  const data = await postJSON("/api/turn", { session_id: sessionId, user_text: text });
  if (data.error) {
    alert(data.error);
    recording = false;
    return;
  }

  // Show salesperson line ONCE (prefer backend-cleaned if present)
  const shown = (data.user_text_clean || text).trim();
  addBubble("sales", shown);

  await handleClientReply(data.client_text, data.done);
  recording = false;
}

async function sendTyped() {
  const text = (typedInput?.value || "").trim();
  if (!text) return;
  typedInput.value = "";

  // barge-in
  if (speaking) stopSpeaking();

  await sendTranscriptToTurn(text);
}

// ----- Client reply handler -----
async function handleClientReply(clientText, done) {
  addBubble("client", clientText);

  setStatus("client speaking...");
  btnTalk.disabled = true;

  await speak(clientText);

  if (done) {
    setStatus("scenario finished — generate report");
    btnTalk.disabled = true;
    btnEnd.disabled = false;
    callActive = false;
    return;
  }

  setStatus(AUTO_LISTEN_AFTER_CLIENT ? "your turn — speak now" : "your turn — push to talk");
  btnEnd.disabled = false;
  callActive = true;
  btnTalk.disabled = false;
}

// ----- Buttons -----
btnStart.addEventListener("click", async () => {
  setStatus("starting...");
  btnStart.disabled = true;

  const data = await postJSON("/api/start", { session_id: sessionId });
  if (data.error) {
    alert(data.error);
    btnStart.disabled = false;
    return;
  }

  // ✅ Make chat panel slightly darker only after scenario begins
  document.body.classList.add("scenario-active");

  btnEnd.disabled = false;
  await handleClientReply(data.client_text, data.done);
});

// Typed send
if (btnSend && typedInput) {
  btnSend.addEventListener("click", async () => {
    if (!callActive) return; // optional: only allow typing during active call
    await sendTyped();
  });

  typedInput.addEventListener("keydown", async (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      if (!callActive) return;
      await sendTyped();
    }
  });
}

// Push-to-talk (record while pressed)
btnTalk.addEventListener("pointerdown", async (e) => {
  e.preventDefault();
  if (!callActive) return;
  if (recording) return;

  recording = true;
  setStatus("listening…");
  btnTalk.classList.add("listening");

  try { btnTalk.setPointerCapture(e.pointerId); } catch (_) {}

  if (holdTimeout) clearTimeout(holdTimeout);
  holdTimeout = setTimeout(async () => {
    if (!recording) return;
    setStatus("stopped (too long) — transcribing…");
    try {
      const blob = await stopRecordingAndGetBlob();
      const stt = await transcribeWithBackend(blob);
      if (stt.error) throw new Error(stt.error);
      await sendTranscriptToTurn(stt.text || "");
    } catch (err) {
      console.error(err);
      setStatus("STT failed — try again");
      recording = false;
    } finally {
      btnTalk.classList.remove("listening");
    }
  }, STT.maxHoldMs);

  try {
    await startRecording();
  } catch (err) {
    console.error(err);
    alert("Mic permission failed. Check browser site settings.");
    setStatus("mic permission failed");
    recording = false;
    btnTalk.classList.remove("listening");
    if (holdTimeout) clearTimeout(holdTimeout);
  }
});

btnTalk.addEventListener("pointerup", async (e) => {
  e.preventDefault();
  if (!recording) return;

  if (holdTimeout) clearTimeout(holdTimeout);

  setStatus("transcribing…");

  try {
    const blob = await stopRecordingAndGetBlob();
    const stt = await transcribeWithBackend(blob);

    if (stt.error) {
      setStatus("STT error — try again");
      console.error("STT error:", stt.error);
      recording = false;
      btnTalk.classList.remove("listening");
      return;
    }

    await sendTranscriptToTurn(stt.text || "");
  } catch (err) {
    console.error(err);
    setStatus("STT failed — try again");
    recording = false;
  } finally {
    btnTalk.classList.remove("listening");
    try { btnTalk.releasePointerCapture(e.pointerId); } catch (_) {}
  }
});

btnTalk.addEventListener("pointercancel", () => {
  if (holdTimeout) clearTimeout(holdTimeout);
  recording = false;
  btnTalk.classList.remove("listening");
});

btnEnd.addEventListener("click", async () => {
  stopSpeaking();
  if (holdTimeout) clearTimeout(holdTimeout);
  recording = false;
  btnTalk.classList.remove("listening");

  // ✅ remove "active" darkening when ending
  document.body.classList.remove("scenario-active");

  setStatus("generating report...");
  const data = await postJSON("/api/end", { session_id: sessionId });

  if (data.error) {
    alert("Report failed: " + data.error);
    setStatus("report failed");
    return;
  }

  const url = data.report_url || ("/report/" + sessionId);
  window.location.href = url;
});
