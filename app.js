// Frog ID web prototype: BirdNET embeddings + our classifier, all in the browser.
const SR = 48000;            // BirdNET's sample rate
const WIN = 3 * SR;          // 3-second windows
const MAX_FILE_WINDOWS = 20; // analyze the first 60 s of a file
const VERSION = "0.1";
const NOT_FROG = "not_frog";
const MONTHS = "JFMAMJJASOND";

const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};

let model, clf, filt, species, speciesBy, samples = [];
let answers = [], toAnswer = [], noiseIdx = [];
let place = store.get("place", null);   // {lat, lng}
let threshold = store.get("threshold", 0.7);

// ---------- loading ----------

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

// Fetch the model once and keep it in Cache Storage, with a progress bar.
async function fetchModel(url, onProgress) {
  const cacheName = "frog-id-model-v1";
  let cache = null;
  try { cache = await caches.open(cacheName); } catch {}
  if (cache) {
    const hit = await cache.match(url);
    if (hit) { onProgress(1); return hit.arrayBuffer(); }
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  const total = Number(r.headers.get("Content-Length")) || 25e6;
  const reader = r.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(Math.min(got / total, 0.99));
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  if (cache) {
    try { await cache.put(url, new Response(buf, { headers: { "Content-Type": "application/octet-stream" } })); } catch {}
  }
  onProgress(1);
  return buf.buffer;
}

async function loadAll() {
  const bar = $("load-bar");
  [clf, filt, species] = await Promise.all([
    fetchJSON("model/frog_classifier.json"),
    fetchJSON("model/filter.json"),
    fetchJSON("model/species.json"),
  ]);
  try { samples = await fetchJSON("samples/samples.json"); } catch { samples = []; }
  speciesBy = Object.fromEntries(species.map((s) => [s.label, s]));
  prepareClassifier();
  renderSpecies();
  renderNear();

  tflite.setWasmPath("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.10/wasm/");
  const buf = await fetchModel("model/birdnet_embed.tflite", (p) => { bar.style.width = `${Math.round(p * 100)}%`; });
  $("load-note").textContent = "Starting the model…";
  model = await tflite.loadTFLiteModel(buf, { numThreads: 1 });
  // Warm up once so the first real window isn't slow.
  embed(new Float32Array(WIN));
  $("loading").hidden = true;
  $("listen").disabled = false;
  $("sample").disabled = samples.length === 0;
}

// ---------- the classifier ----------

function prepareClassifier() {
  const frogs = clf.classes.filter((c) => !c.startsWith("noise_"));
  answers = [...frogs, NOT_FROG];
  toAnswer = clf.classes.map((c) => (c.startsWith("noise_") ? answers.length - 1 : answers.indexOf(c)));
  noiseIdx = clf.classes.map((c, k) => (c.startsWith("noise_") ? k : -1)).filter((k) => k >= 0);
  // Flatten weights for speed.
  clf.W = clf.coef.map((row) => Float32Array.from(row));
  clf.m = Float32Array.from(clf.mean);
  clf.s = Float32Array.from(clf.scale);
}

function embed(window48k) {
  return tf.tidy(() => {
    const out = model.predict(tf.tensor(window48k, [1, WIN]));
    const t = out instanceof tf.Tensor ? out : Object.values(out)[0];
    return t.dataSync().slice();
  });
}

// Class probabilities (softmax) for one embedding.
function classProbs(e) {
  const n = e.length;
  const z = new Float32Array(n);
  for (let i = 0; i < n; i++) z[i] = (e[i] - clf.m[i]) / clf.s[i];
  const logits = clf.W.map((w, k) => {
    let s = clf.intercept[k];
    for (let i = 0; i < n; i++) s += w[i] * z[i];
    return s;
  });
  const mx = Math.max(...logits);
  const ex = logits.map((l) => Math.exp(l - mx));
  const sum = ex.reduce((a, b) => a + b, 0);
  return ex.map((x) => x / sum);
}

// Merge noise classes into "not a frog", then (live only) apply the location/season filter.
function toAnswers(p, live = true) {
  const a = new Array(answers.length).fill(0);
  p.forEach((x, k) => { a[toAnswer[k]] += x; });
  const ok = live ? filterSet() : null;
  if (ok) {
    for (let j = 0; j < answers.length - 1; j++) if (!ok.has(answers[j])) a[j] = 0;
    const s = a.reduce((x, y) => x + y, 0);
    if (s > 0) for (let j = 0; j < a.length; j++) a[j] /= s;
  }
  return a;
}

function noiseKind(p) {
  let best = -1, bp = -1;
  for (const k of noiseIdx) if (p[k] > bp) { bp = p[k]; best = k; }
  return clf.common_names[best];
}

// ---------- the location + season filter ----------

function cellOf(lat, lng) {
  const r = Math.floor((lat - filt.lat0) / filt.cell);
  const c = Math.floor((lng - filt.lng0) / filt.cell);
  return r >= 0 && r < filt.rows && c >= 0 && c < filt.cols ? [r, c] : null;
}

function allowedSet(lat, lng, month) {
  let ok = new Set(filt.species);
  if (lat != null) {
    const cell = cellOf(lat, lng);
    if (cell) ok = new Set(filt.cells[cell[0]][cell[1]].map((k) => filt.species[k]));
  }
  if (month) ok = new Set([...ok].filter((l) => filt.months[l].on[month - 1]));
  return ok;
}

function filterSet() {
  if (!$("use-filter").checked) return null;
  const month = new Date().getMonth() + 1;
  return allowedSet(place ? place.lat : null, place ? place.lng : null, month);
}

function seasonBar(label) {
  const on = filt.months[label].on;
  const now = new Date().getMonth();
  return `<span class="season" aria-label="Calls in ${on.map((x, i) => (x ? "JanFebMarAprMayJunJulAugSepOctNovDec".slice(i * 3, i * 3 + 3) : null)).filter(Boolean).join(", ")}">${
    on.map((x, i) => `<i class="${x ? "on" : ""}${i === now ? " now" : ""}">${MONTHS[i]}</i>`).join("")}</span>`;
}

// ---------- rendering ----------

function nameOf(label) {
  return label === NOT_FROG ? "Not a frog" : (speciesBy[label] ? speciesBy[label].name : label);
}

function speciesRow(s, extra = "") {
  return `<li><button class="sp" data-label="${s.label}">
    <span class="sp-name">${s.name}<em>${s.scientific}</em></span>${extra}${seasonBar(s.label)}</button></li>`;
}

function renderSpecies() {
  const inModel = new Set(clf.classes);
  $("species-list").innerHTML = species.map((s) =>
    speciesRow(s, inModel.has(s.label) ? "" : `<span class="pill warn">not enough data yet</span>`)).join("");
}

function renderNear() {
  const month = new Date().getMonth() + 1;
  const monthName = new Date().toLocaleString("en-US", { month: "long" });
  const ok = allowedSet(place ? place.lat : null, place ? place.lng : null, month);
  const list = species.filter((s) => ok.has(s.label));
  let status;
  if (!place) status = `No location yet, so this is every frog that calls in Texas in ${monthName}.`;
  else if (!cellOf(place.lat, place.lng)) status = `You're outside Texas, so only the season is used.`;
  else status = `${list.length} of ${species.length} frogs are possible near you in ${monthName}.`;
  $("near-status").textContent = status;
  $("near-list").innerHTML = list.map((s) => speciesRow(s)).join("") ||
    `<li class="muted">None expected right now. Frogs mostly call on warm, rainy nights.</li>`;
  const fn = $("filter-note");
  fn.textContent = $("use-filter").checked
    ? `Filter on: only ${list.length} frog${list.length === 1 ? "" : "s"} possible ${place ? "near you" : "in Texas"} in ${monthName}. Testing with a recording played from a speaker? Tap to turn it off.`
    : "Filter off: any frog can be suggested. Tap to change.";
  $("where").hidden = !place;
  if (place) $("where").textContent = `📍 ${place.lat.toFixed(2)}, ${place.lng.toFixed(2)}`;
}

function openSpecies(label) {
  const s = speciesBy[label];
  if (!s) return;
  const taxon = s.inat_taxon;
  const link = taxon ? `https://www.inaturalist.org/taxa/${taxon}` : `https://www.inaturalist.org/search?q=${encodeURIComponent(s.scientific)}`;
  const sounds = taxon ? `https://www.inaturalist.org/observations?taxon_id=${taxon}&sounds=true&place_id=18&quality_grade=research` : link;
  const counts = filt.months[label].counts;
  const max = Math.max(1, ...counts);
  $("sheet-content").innerHTML = `
    <h2>${s.name}</h2><p class="muted"><em>${s.scientific}</em></p>
    <div class="hist">${counts.map((c, i) => `<div><span style="height:${Math.round(60 * c / max)}px"></span><b>${MONTHS[i]}</b></div>`).join("")}</div>
    <p class="muted small">${filt.months[label].source === "calling records" ? "Texas recordings of it calling, by month." : filt.months[label].source === "southern US calling records" ? "Recordings of it calling across the southern US, by month (few in Texas)." : "Texas sightings by month (very few recordings)."}</p>
    <a class="ghost block" href="${link}" target="_blank" rel="noopener">Photos and info on iNaturalist</a>
    <a class="ghost block" href="${sounds}" target="_blank" rel="noopener">Hear Texas recordings</a>`;
  $("sheet").hidden = false;
}

// ---------- results ----------

const heard = new Map(); // label -> {best, count, last}

function showNow(a, p, label) {
  let best = 0;
  for (let j = 1; j < a.length; j++) if (a[j] > a[best]) best = j;
  const conf = a[best];
  const meter = $("now-meter");
  meter.hidden = false;
  meter.firstElementChild.style.width = `${Math.round(conf * 100)}%`;
  if (answers[best] === NOT_FROG) {
    $("now-title").textContent = "No frog";
    $("now-sub").textContent = `Sounds like: ${noiseKind(p).toLowerCase()} (${Math.round(conf * 100)}%)`;
    meter.className = "meter quiet";
  } else if (conf < threshold) {
    $("now-title").textContent = "Not sure";
    $("now-sub").textContent = `Maybe ${nameOf(answers[best])} (${Math.round(conf * 100)}%)`;
    meter.className = "meter quiet";
  } else {
    $("now-title").textContent = nameOf(answers[best]);
    $("now-sub").textContent = `${Math.round(conf * 100)}% sure${label ? ` · ${label}` : ""}`;
    meter.className = "meter";
    if (label) return; // a recording, not live listening
    const h = heard.get(answers[best]) || { best: 0, count: 0 };
    h.best = Math.max(h.best, conf);
    h.count += 1;
    h.last = new Date();
    heard.set(answers[best], h);
    renderHeard(answers[best]);
  }
}

function renderHeard(flash) {
  $("heard-h").hidden = heard.size === 0;
  $("heard").innerHTML = [...heard.entries()]
    .sort((x, y) => y[1].last - x[1].last)
    .map(([l, h]) => `<li class="${l === flash ? "flash" : ""}"><button class="sp" data-label="${l}">
      <span class="sp-name">${nameOf(l)}<em>${h.count} × · best ${Math.round(h.best * 100)}% · ${h.last.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}</em></span></button></li>`)
    .join("");
}

// ---------- live listening ----------

let audio = null; // {ctx, stream, proc, analyser, buf, n, busy}

async function startListening() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
  });
  let ctx;
  try { ctx = new AudioContext({ sampleRate: SR }); } catch { ctx = new AudioContext(); }
  await ctx.resume();
  const src = ctx.createMediaStreamSource(stream);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 2048;
  analyser.smoothingTimeConstant = 0;
  src.connect(analyser);
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  const rate = ctx.sampleRate;
  const need = Math.round(3 * rate);
  audio = { ctx, stream, proc, analyser, buf: new Float32Array(need), n: 0, busy: false };
  proc.onaudioprocess = (ev) => {
    const x = ev.inputBuffer.getChannelData(0);
    let i = 0;
    while (i < x.length) {
      const take = Math.min(x.length - i, need - audio.n);
      audio.buf.set(x.subarray(i, i + take), audio.n);
      audio.n += take;
      i += take;
      if (audio.n === need) {
        const win = rate === SR ? audio.buf.slice() : resample(audio.buf, rate, SR);
        audio.n = 0;
        if (!audio.busy) {
          audio.busy = true;
          setTimeout(() => {
            try {
              const p = classProbs(embed(win));
              showNow(toAnswers(p), p);
            } finally { if (audio) audio.busy = false; }
          }, 0);
        }
      }
    }
  };
  src.connect(proc);
  proc.connect(ctx.destination);
  $("listen").textContent = "Stop";
  $("listen").classList.add("on");
  $("now-title").textContent = "Listening…";
  $("now-sub").textContent = "First answer in 3 seconds.";
  drawSpectro();
}

function stopListening() {
  if (!audio) return;
  audio.proc.disconnect();
  audio.stream.getTracks().forEach((t) => t.stop());
  audio.ctx.close();
  audio = null;
  $("listen").textContent = "Listen";
  $("listen").classList.remove("on");
}

function resample(x, from, to) {
  const n = Math.round(x.length * to / from);
  const out = new Float32Array(n);
  const step = from / to;
  for (let i = 0; i < n; i++) {
    const t = i * step, k = Math.floor(t), f = t - k;
    out[i] = (x[k] || 0) * (1 - f) + (x[k + 1] || 0) * f;
  }
  return out;
}

// Scrolling spectrogram, 0-8 kHz.
function drawSpectro() {
  const cv = $("spectro"), g = cv.getContext("2d");
  const bins = new Uint8Array(audio.analyser.frequencyBinCount);
  const hz = audio.ctx.sampleRate / 2 / bins.length;
  const top = Math.min(bins.length, Math.floor(8000 / hz));
  const step = () => {
    if (!audio) return;
    audio.analyser.getByteFrequencyData(bins);
    g.drawImage(cv, -2, 0);
    for (let y = 0; y < cv.height; y++) {
      const b = Math.floor((1 - y / cv.height) * top);
      const v = bins[b] / 255;
      g.fillStyle = `rgb(${Math.round(30 + 200 * v * v)}, ${Math.round(40 + 215 * v)}, ${Math.round(35 + 90 * v)})`;
      g.fillRect(cv.width - 2, y, 2, 1);
    }
    requestAnimationFrame(step);
  };
  g.fillStyle = "#0b120e";
  g.fillRect(0, 0, cv.width, cv.height);
  step();
}

// ---------- recordings ----------

async function decodeTo48k(arrayBuffer) {
  const tmp = new (window.AudioContext || window.webkitAudioContext)();
  const decoded = await new Promise((res, rej) => tmp.decodeAudioData(arrayBuffer, res, rej));
  tmp.close();
  const secs = Math.min(decoded.duration, MAX_FILE_WINDOWS * 3);
  const off = new OfflineAudioContext(1, Math.ceil(secs * SR), SR);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return rendered.getChannelData(0);
}

async function analyzeBuffer(arrayBuffer, title) {
  const box = $("file-result");
  box.hidden = false;
  box.innerHTML = `<h2>${title}</h2><p class="muted">Analyzing…</p>`;
  let x;
  try { x = await decodeTo48k(arrayBuffer); } catch (e) {
    box.innerHTML = `<h2>${title}</h2><p>Couldn't read that file (${e.message || e}). Try an m4a, mp3 or wav.</p>`;
    return;
  }
  const rows = [], mean = new Array(answers.length).fill(0);
  for (let s = 0; s + WIN / 2 <= x.length && rows.length < MAX_FILE_WINDOWS; s += WIN) {
    const w = new Float32Array(WIN);
    w.set(x.subarray(s, s + WIN));
    const p = classProbs(embed(w));
    const a = toAnswers(p, false); // a file's place and date are unknown, so no filter
    a.forEach((v, j) => { mean[j] += v; });
    rows.push({ t: s / SR, a, p });
    await new Promise((r) => setTimeout(r, 0)); // keep the page responsive
  }
  if (!rows.length) { box.innerHTML = `<h2>${title}</h2><p>That recording is too short (needs 1.5 s or more).</p>`; return; }
  mean.forEach((v, j) => { mean[j] = v / rows.length; });
  const order = mean.map((v, j) => j).sort((i, j) => mean[j] - mean[i]);
  const best = order[0];
  const verdict = answers[best] === NOT_FROG ? "No frog" : mean[best] < threshold ? `Not sure: maybe ${nameOf(answers[best])}` : nameOf(answers[best]);
  showNow(mean, rows[0].p, "recording");
  box.innerHTML = `<h2>${title}</h2>
    <p class="verdict">${verdict} <span class="muted">${Math.round(mean[best] * 100)}%</span></p>
    <ol class="top3">${order.slice(0, 3).map((j) => `<li>${nameOf(answers[j])} <span class="muted">${Math.round(mean[j] * 100)}%</span></li>`).join("")}</ol>
    <p class="muted small">The location and season filter is only used when listening live.</p>
    <h3>Every 3 seconds</h3>
    <ul class="windows">${rows.map((r) => {
      let b = 0;
      for (let j = 1; j < r.a.length; j++) if (r.a[j] > r.a[b]) b = j;
      const nm = answers[b] === NOT_FROG ? `no frog (${noiseKind(r.p).toLowerCase()})` : nameOf(answers[b]);
      return `<li><b>${r.t.toFixed(0)}s</b> ${nm} <span class="muted">${Math.round(r.a[b] * 100)}%</span></li>`;
    }).join("")}</ul>`;
}

// ---------- wiring ----------

function setupUI() {
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("active", x === b));
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${b.dataset.tab}`));
    window.scrollTo(0, 0);
  }));
  document.body.addEventListener("click", (e) => {
    const sp = e.target.closest(".sp");
    if (sp) openSpecies(sp.dataset.label);
  });
  $("sheet").addEventListener("click", (e) => {
    if (e.target.id === "sheet" || e.target.classList.contains("sheet-close")) $("sheet").hidden = true;
  });

  $("listen").addEventListener("click", async () => {
    if (audio) return stopListening();
    try { await startListening(); } catch (e) {
      $("now-title").textContent = "Can't use the microphone";
      $("now-sub").textContent = e.name === "NotAllowedError"
        ? "Allow microphone access for this site in your browser settings, then try again."
        : String(e.message || e);
    }
  });

  $("file").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (!f || !model) return;
    if (sampleAudio) { sampleAudio.pause(); sampleAudio = null; }
    await analyzeBuffer(await f.arrayBuffer(), f.name);
    e.target.value = "";
  });

  let sampleAudio = null;
  $("sample").addEventListener("click", async () => {
    const s = samples[Math.floor(Math.random() * samples.length)];
    // Start playing right away, inside the tap: iPhones only allow sound that a tap starts.
    if (sampleAudio) sampleAudio.pause();
    const player = sampleAudio = new Audio(`samples/${s.file}`);
    player.controls = true;
    player.className = "player";
    player.play().catch(() => {}); // if it's blocked, the controls below still work
    const r = await fetch(`samples/${s.file}`);
    await analyzeBuffer(await r.arrayBuffer(), `Sample: ${s.name}`);
    if (player !== sampleAudio) return; // another sample was tapped meanwhile
    $("file-result").querySelector("h2").after(player);
    $("file-result").insertAdjacentHTML("beforeend",
      `<p class="muted small">Recording: ${s.attribution} (<a href="${s.url}" target="_blank" rel="noopener">iNaturalist</a>).
      The model was trained on this recording, so it's an easy test: it shows the app works, not how accurate it is.</p>`);
  });

  $("use-filter").checked = store.get("useFilter", true);
  $("use-filter").addEventListener("change", () => { store.set("useFilter", $("use-filter").checked); renderNear(); });
  $("locate").addEventListener("click", () => {
    $("near-status").textContent = "Finding you…";
    navigator.geolocation.getCurrentPosition((pos) => {
      // Rounded to ~1 km; it's only used on this phone.
      place = { lat: Math.round(pos.coords.latitude * 100) / 100, lng: Math.round(pos.coords.longitude * 100) / 100 };
      store.set("place", place);
      renderNear();
    }, (err) => { $("near-status").textContent = `Couldn't get your location (${err.message}).`; },
    { enableHighAccuracy: false, timeout: 15000, maximumAge: 3600e3 });
  });

  $("filter-note").addEventListener("click", () => document.querySelector('.tabs button[data-tab="near"]').click());
  $("threshold").value = String(threshold);
  $("threshold").addEventListener("change", () => { threshold = Number($("threshold").value); store.set("threshold", threshold); });
  $("version").textContent = `Version ${VERSION}`;
}

setupUI();
loadAll().catch((e) => {
  console.error(e);
  $("load-note").textContent = `Couldn't load the model: ${e.message || e}`;
});

// For testing from the console.
window.frogid = { analyzeBuffer, classProbs, embed, toAnswers, get answers() { return answers; } };
