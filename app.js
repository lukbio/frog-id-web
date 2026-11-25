// Frog ID web prototype: BirdNET embeddings + our classifier, all in the browser.
const SR = 48000;            // BirdNET's sample rate
const WIN = 3 * SR;          // 3-second windows
const MAX_FILE_WINDOWS = 20; // analyze the first 60 s of a file
const VERSION = "0.6";
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
let useFilter = store.get("useFilter", true);
let showSci = store.get("showSci", false);
let life = store.get("life", {});       // label -> ISO date it was added
let photos = {};                        // label -> {url, attribution} (CC-licensed iNaturalist photos)

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
  renderAll();
  loadPhotos();
  route();

  tflite.setWasmPath("vendor/");
  const buf = await fetchModel("model/birdnet_embed.tflite", (p) => { bar.style.width = `${Math.round(p * 100)}%`; });
  $("load-note").textContent = "Starting the model…";
  model = await tflite.loadTFLiteModel(buf, { numThreads: 1 });
  // Warm up once so the first real window isn't slow.
  embed(new Float32Array(WIN));
  $("loading").hidden = true;
  $("listen").disabled = false;
  $("listen-home").disabled = false;
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

// Class scores for one embedding: an independent 0-1 score per class
// ("sigmoid", so several frogs can score high at once), or a softmax for older models.
function classProbs(e) {
  const n = e.length;
  const z = new Float32Array(n);
  for (let i = 0; i < n; i++) z[i] = (e[i] - clf.m[i]) / clf.s[i];
  const logits = clf.W.map((w, k) => {
    let s = clf.intercept[k];
    for (let i = 0; i < n; i++) s += w[i] * z[i];
    return s;
  });
  if (clf.output === "sigmoid") return logits.map((l) => 1 / (1 + Math.exp(-l)));
  const mx = Math.max(...logits);
  const ex = logits.map((l) => Math.exp(l - mx));
  const sum = ex.reduce((a, b) => a + b, 0);
  return ex.map((x) => x / sum);
}

// Merge noise classes into "not a frog" (their highest score, or their sum for
// a softmax), then (live only) drop frogs the location/season filter rules out.
function toAnswers(p, live = true) {
  const sig = clf.output === "sigmoid";
  const a = new Array(answers.length).fill(0);
  p.forEach((x, k) => { const j = toAnswer[k]; a[j] = sig ? Math.max(a[j], x) : a[j] + x; });
  const ok = live ? filterSet() : null;
  if (ok) {
    for (let j = 0; j < answers.length - 1; j++) if (!ok.has(answers[j])) a[j] = 0;
    if (!sig) {
      const s = a.reduce((x, y) => x + y, 0);
      if (s > 0) for (let j = 0; j < a.length; j++) a[j] /= s;
    }
  }
  return a;
}

// Frogs at or above the threshold, best first: [[answerIndex, score], ...]
function calling(a) {
  const out = [];
  for (let j = 0; j < answers.length - 1; j++) if (a[j] >= threshold) out.push([j, a[j]]);
  return out.sort((x, y) => y[1] - x[1]);
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
  if (!useFilter) return null;
  return likelyNow();
}

// Frogs possible here (or anywhere in the grid) this month, whatever the filter setting.
function likelyNow() {
  const month = new Date().getMonth() + 1;
  return allowedSet(place ? place.lat : null, place ? place.lng : null, month);
}

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const MONTH_NAMES = "JanFebMarAprMayJunJulAugSepOctNovDec";
const monthName = () => new Date().toLocaleString("en-US", { month: "long" });
const placeName = () => (place ? (cellOf(place.lat, place.lng) ? "near you" : "in East Texas") : "in East Texas");

// Month bars: height follows how many calling records there are each month,
// dark where it calls, with today's month marked.
function monthBars(label) {
  const m = filt.months[label];
  const max = Math.max(1, ...m.counts);
  const now = new Date().getMonth();
  const said = m.on.map((x, i) => (x ? MONTH_NAMES.slice(i * 3, i * 3 + 3) : null)).filter(Boolean).join(", ");
  return `<span class="months" role="img" aria-label="Calls in ${said}">${m.counts.map((c, i) =>
    `<i class="${m.on[i] ? "on" : ""}${i === now ? " cur" : ""}"><span style="height:${m.on[i] ? Math.max(5, Math.round(18 * Math.sqrt(c / max))) : 3}px"></span></i>`).join("")}</span>
    <span class="months-labels" aria-hidden="true">${[...MONTHS].map((x) => `<b>${x}</b>`).join("")}</span>`;
}

// A photo, or a plain tile with a frog drawn on it (offline, or no CC-licensed photo).
function photo(label, size = "medium") {
  const ph = photos[label];
  const frog = `<svg class="ph-icon"><use href="#i-frog"/></svg>`;
  if (!ph) return frog;
  const url = size === "medium" ? ph.url : ph.url.replace(/\/medium\./, `/${size}.`);
  return `${frog}<img src="${esc(url)}" alt="" loading="lazy" onerror="this.remove()">`;
}

// ---------- rendering ----------

function nameOf(label) {
  return label === NOT_FROG ? "Not a frog" : (speciesBy[label] ? speciesBy[label].name : label);
}

function speciesRow(s, extra = "") {
  return `<li><button class="sp" data-label="${s.label}" type="button">
    <span class="thumb ph">${photo(s.label, "small")}</span>
    <span class="sp-body">
      <span class="sp-name">${s.name}${showSci ? `<em>${s.scientific}</em>` : ""}</span>${extra}
      <span>${monthBars(s.label)}</span>
    </span></button></li>`;
}

function renderAll() {
  renderHome();
  renderExplore();
  renderLife();
  renderSettings();
}

function renderHome() {
  const ok = likelyNow();
  const list = species.filter((s) => ok.has(s.label));
  $("calling-count").textContent = `${list.length} calling in ${monthName()} ${placeName()}`;
  // Frog of the day: one of this month's frogs (any frog if none), the same all day.
  const pool = (list.length ? list : species).filter((s) => photos[s.label]);
  const from = pool.length ? pool : (list.length ? list : species);
  const d = new Date();
  const day = Math.floor((d - new Date(d.getFullYear(), 0, 0)) / 864e5);
  const s = from[day % from.length];
  $("hero").dataset.label = s.label;
  $("hero-name").textContent = s.name;
  $("hero-img").innerHTML = photo(s.label);
  const fn = $("filter-note");
  fn.textContent = useFilter
    ? `Filter on: only the ${list.length} frog${list.length === 1 ? "" : "s"} that call ${placeName()} in ${monthName()} can be suggested. Testing with a recording played from a speaker? Tap to change.`
    : "Filter off: any frog can be suggested. Tap to change.";
}

function renderExplore() {
  const ok = likelyNow();
  const q = $("search").value.trim().toLowerCase();
  const inModel = new Set(clf.classes);
  const match = (s) => !q || s.name.toLowerCase().includes(q) || s.scientific.toLowerCase().includes(q);
  const likely = species.filter((s) => ok.has(s.label) && match(s));
  const other = species.filter((s) => !ok.has(s.label) && match(s));
  const extra = (s) => (inModel.has(s.label) ? "" : `<span class="pill warn">can't identify yet: too few recordings</span>`);
  let status = `${ok.size} likely in ${monthName()} ${placeName()}`;
  if (place && !cellOf(place.lat, place.lng)) status += " (you're outside the map, so only the season is used)";
  $("near-status").textContent = status;
  $("explore-list").innerHTML =
    likely.map((s) => speciesRow(s, extra(s))).join("") +
    (other.length ? `<li class="list-h">Not calling ${placeName()} right now</li>` + other.map((s) => speciesRow(s, extra(s))).join("") : "") ||
    `<li class="muted">No frog matches “${esc(q)}”.</li>`;
}

function renderLife() {
  const mine = Object.entries(life).sort((a, b) => b[1].localeCompare(a[1]));
  $("life-sub").textContent = mine.length
    ? `${mine.length} of ${species.length} East Texas frogs`
    : "Add your first frog!";
  $("life-list").innerHTML = mine.filter(([l]) => speciesBy[l]).map(([l, when]) =>
    speciesRow(speciesBy[l], `<span class="sp-meta">Added ${new Date(when).toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" })}</span>`)).join("");
  const ok = likelyNow();
  const todo = species.filter((s) => ok.has(s.label) && !life[s.label]);
  $("find-sub").textContent = `Likely in ${monthName()} ${placeName()}`;
  $("find-grid").innerHTML = todo.map((s) => `<button class="cell" data-label="${s.label}" type="button">
      <span class="cell-img ph">${photo(s.label)}</span>
      <span class="cell-cap"><b>${s.name}</b><em>${s.scientific}</em></span></button>`).join("") ||
    `<p class="muted">You've found every frog that calls here this month.</p>`;
}

function renderSettings() {
  $("use-filter").checked = useFilter;
  $("show-sci").checked = showSci;
  $("place-value").textContent = place ? `${place.lat.toFixed(2)}, ${place.lng.toFixed(2)} ›` : "Not set ›";
}

function locate() {
  $("place-value").textContent = "Finding you…";
  navigator.geolocation.getCurrentPosition((pos) => {
    // Rounded to ~1 km; it's only used on this phone.
    place = { lat: Math.round(pos.coords.latitude * 100) / 100, lng: Math.round(pos.coords.longitude * 100) / 100 };
    store.set("place", place);
    renderAll();
  }, (err) => {
    $("place-value").textContent = "Couldn't get it";
    alert(`Couldn't get your location (${err.message}).`);
  }, { enableHighAccuracy: false, timeout: 15000, maximumAge: 3600e3 });
}

// Photos: one iNaturalist API call, keeping only Creative Commons photos, saved on the phone.
async function loadPhotos() {
  const saved = store.get("photos", null);
  if (saved && saved.v === 1 && Date.now() - saved.t < 30 * 864e5) { photos = saved.data; renderAll(); return; }
  const ids = species.map((s) => s.inat_taxon).filter(Boolean);
  try {
    const r = await fetch(`https://api.inaturalist.org/v1/taxa/${ids.join(",")}?per_page=${ids.length}`);
    if (!r.ok) throw new Error(r.status);
    const byTaxon = {};
    for (const t of (await r.json()).results) {
      const p = t.default_photo;
      if (p && p.license_code && p.medium_url) byTaxon[t.id] = { url: p.medium_url, attribution: p.attribution };
    }
    photos = {};
    for (const s of species) if (byTaxon[s.inat_taxon]) photos[s.label] = byTaxon[s.inat_taxon];
    store.set("photos", { v: 1, t: Date.now(), data: photos });
    renderAll();
  } catch (e) {
    if (saved && saved.data) { photos = saved.data; renderAll(); } // offline: keep the old ones
  }
}

// ---------- a frog's page (#frog=label) ----------

// Simplified Texas outline (lng, lat), for the range map.
const TEXAS = [[-106.62, 32.0], [-103.06, 32.0], [-103.04, 36.5], [-100.0, 36.5], [-100.0, 34.56], [-99.2, 34.37], [-98.1, 34.13],
  [-97.1, 33.73], [-96.3, 33.76], [-95.3, 33.87], [-94.48, 33.64], [-94.04, 33.55], [-94.04, 32.0], [-93.8, 31.5], [-93.55, 31.0],
  [-93.57, 30.4], [-93.72, 30.06], [-93.84, 29.7], [-94.5, 29.5], [-94.9, 29.3], [-95.3, 28.9], [-96.0, 28.6], [-96.6, 28.3],
  [-97.0, 27.9], [-97.3, 27.4], [-97.4, 26.8], [-97.15, 25.95], [-97.5, 25.9], [-98.0, 26.05], [-98.5, 26.2], [-99.1, 26.5],
  [-99.4, 27.0], [-99.5, 27.5], [-100.0, 28.1], [-100.3, 28.6], [-100.6, 29.1], [-101.0, 29.5], [-101.4, 29.77], [-102.0, 29.8],
  [-102.4, 29.77], [-102.7, 29.6], [-102.9, 29.2], [-103.2, 28.98], [-103.6, 29.16], [-104.0, 29.3], [-104.5, 29.7], [-104.7, 30.2],
  [-105.0, 30.7], [-105.6, 31.1], [-106.2, 31.5], [-106.5, 31.78]];
const ABOUT_DAYS = 30;
let frogPlayer = null;   // the one Audio element playing a call
let pageScroll = 0;      // where the list was, to go back to it

// Extra photos, a description and recordings for one frog, from iNaturalist, saved for 30 days.
async function frogDetails(s) {
  const key = `taxon:${s.inat_taxon}`;
  const saved = store.get(key, null);
  if (saved && saved.v === 1 && Date.now() - saved.t < ABOUT_DAYS * 864e5) return saved;
  if (!s.inat_taxon) return null;
  try {
    const [t, o] = await Promise.all([
      fetch(`https://api.inaturalist.org/v1/taxa/${s.inat_taxon}`).then((r) => r.json()),
      fetch(`https://api.inaturalist.org/v1/observations?taxon_id=${s.inat_taxon}&sounds=true&place_id=18&quality_grade=research&order_by=votes&per_page=12`).then((r) => r.json()),
    ]);
    const taxon = t.results[0] || {};
    const pics = (taxon.taxon_photos || []).map((x) => x.photo)
      .filter((p) => p && p.license_code && p.medium_url).slice(0, 8)
      .map((p) => ({ url: p.medium_url.replace(/\/medium\./, "/large."), attribution: p.attribution }));
    const sounds = [];
    for (const ob of o.results || []) {
      const snd = (ob.sounds || []).find((x) => x.license_code && x.file_url);
      if (snd) sounds.push({ url: snd.file_url, attribution: snd.attribution, date: ob.observed_on || "", where: ob.place_guess || "", obs: ob.id });
      if (sounds.length === 3) break;
    }
    const text = taxon.wikipedia_summary ? new DOMParser().parseFromString(taxon.wikipedia_summary, "text/html").body.textContent.trim() : "";
    const d = { v: 1, t: Date.now(), pics, sounds, text, wiki: taxon.wikipedia_url || "" };
    store.set(key, d);
    return d;
  } catch {
    return saved; // offline: whatever we had, even if old
  }
}

function rangeMap(label) {
  const k = filt.species.indexOf(label);
  const kx = Math.cos(31 * Math.PI / 180) * 20, ky = 20; // px per degree
  const W = filt.cols * filt.cell * kx, H = filt.rows * filt.cell * ky;
  const X = (lng) => (lng - filt.lng0) * kx, Y = (lat) => H - (lat - filt.lat0) * ky;
  let cells = "";
  for (let r = 0; r < filt.rows; r++) for (let c = 0; c < filt.cols; c++) {
    if (filt.cells[r][c].includes(k)) {
      cells += `M${(c * filt.cell * kx).toFixed(1)} ${(H - (r + 1) * filt.cell * ky).toFixed(1)}h${(filt.cell * kx).toFixed(1)}v${(filt.cell * ky).toFixed(1)}h-${(filt.cell * kx).toFixed(1)}z`;
    }
  }
  const outline = TEXAS.map(([lng, lat]) => `${X(lng).toFixed(1)},${Y(lat).toFixed(1)}`).join(" ");
  const me = place && cellOf(place.lat, place.lng)
    ? `<circle class="me" cx="${X(place.lng).toFixed(1)}" cy="${Y(place.lat).toFixed(1)}" r="6"/>` : "";
  return `<svg class="range" viewBox="0 0 ${W.toFixed(0)} ${H.toFixed(0)}" role="img" aria-label="Map of where ${speciesBy[label].name} has been seen in and around Texas">
    <path class="cells" d="${cells}" shape-rendering="crispEdges"/><polygon class="tx" points="${outline}"/>${me}</svg>`;
}

function soundRow(src, title, sub, i) {
  return `<li><button class="call" data-src="${esc(src)}" type="button" aria-label="Play ${esc(title)}">
    <span class="play"><svg><use href="#i-play"/></svg></span>
    <span><b>${esc(title)}</b><span class="muted small">${sub}</span></span></button></li>`;
}

function renderFrog(label, d) {
  const s = speciesBy[label];
  const ok = likelyNow().has(label);
  const m = filt.months[label];
  const max = Math.max(1, ...m.counts);
  const nowM = new Date().getMonth();
  const pics = d && d.pics && d.pics.length ? d.pics : (photos[label] ? [{ url: photos[label].url, attribution: photos[label].attribution }] : []);
  const sample = samples.find((x) => x.label === label);
  const calls = [];
  if (sample) calls.push(soundRow(`samples/${sample.file}`, "Calls", `Works offline · ${esc(sample.attribution)}`));
  for (const x of (d && d.sounds) || []) {
    const when = x.date ? new Date(`${x.date}T12:00`).toLocaleDateString([], { month: "short", year: "numeric" }) : "";
    calls.push(soundRow(x.url, "Calls", [when, x.where && esc(x.where.split(",").slice(-2).join(",").trim())].filter(Boolean).join(" · ") + `<br>${esc(x.attribution)}`));
  }
  const where = place ? (cellOf(place.lat, place.lng) ? "near you" : "in East Texas") : "in East Texas";
  const taxon = s.inat_taxon;
  const inatSounds = taxon ? `https://www.inaturalist.org/observations?taxon_id=${taxon}&sounds=true&place_id=18&quality_grade=research` : "#";
  const added = life[label];
  $("frog-title").textContent = s.name;
  $("frog-page").innerHTML = `
    <div class="gallery" id="gallery">${pics.length ? pics.map((p, i) =>
      `<div class="slide"><img src="${esc(p.url)}" alt="${i ? "" : esc(s.name)}" ${i ? 'loading="lazy"' : ""} onerror="this.remove()"></div>`).join("")
      : `<div class="slide ph"><svg class="ph-icon"><use href="#i-frog"/></svg></div>`}</div>
    <div class="gallery-cap"><span id="pic-credit">${pics.length ? esc(pics[0].attribution) : (navigator.onLine ? "No photo yet" : "Photos load when you're online")}</span><span id="pic-count">${pics.length > 1 ? `1 of ${pics.length}` : ""}</span></div>

    <div class="frog-body">
      <div class="frog-name">
        <div><h1>${s.name}</h1><p class="sci">${s.scientific}</p></div>
        ${calls.length ? `<button class="round" id="frog-hear" type="button" aria-label="Play its call"><svg><use href="#i-speaker"/></svg></button>` : ""}
      </div>
      ${clf.classes.includes(label) ? "" : `<p class="pill warn">The app can't identify this one yet: too few recordings.</p>`}
      <hr>
      <h2 class="likely">${ok ? `<span class="ok">Likely</span> calling ${where} in ${monthName()}` : `<span class="muted">Not expected</span> ${where} in ${monthName()}`}</h2>
      <div class="hist big">${m.counts.map((c, i) => `<div class="${m.on[i] ? "on" : ""}${i === nowM ? " cur" : ""}"><span style="height:${Math.max(m.on[i] ? 6 : 3, Math.round(100 * c / max))}px"></span><b>${MONTHS[i]}</b></div>`).join("")}</div>
      <p class="muted small">${m.source === "calling records" ? `Recordings of it calling in Texas, by month (${m.counts.reduce((a, b) => a + b, 0)} in all).` : m.source === "southern US calling records" ? "Recordings of it calling across the southern US, by month (few in Texas)." : "Texas sightings by month (very few recordings)."}</p>

      ${d && d.text ? `<h2 class="section-h">About</h2><p class="about-text">${esc(d.text)}</p>
        <p class="muted small">From <a href="${esc(d.wiki)}" target="_blank" rel="noopener">Wikipedia</a>, CC BY-SA.</p>` : ""}

      <h2 class="section-h">Sounds</h2>
      ${calls.length ? `<ul class="calls">${calls.join("")}</ul>` : `<p class="muted">${navigator.onLine ? "No recordings to play yet." : "Recordings load when you're online."}</p>`}
      <a class="btn outline" href="${inatSounds}" target="_blank" rel="noopener">Explore all Texas recordings</a>

      <h2 class="section-h">Range</h2>
      ${rangeMap(label)}
      <div class="legend"><span><i class="sw seen"></i>Seen within this square (iNaturalist)</span>${place && cellOf(place.lat, place.lng) ? `<span><i class="sw you"></i>You</span>` : ""}</div>
      <p class="muted small">Each square is half a degree, about 50 km. The outline is Texas.</p>
      <a class="btn outline" href="https://www.inaturalist.org/taxa/${taxon}" target="_blank" rel="noopener">Photos and info on iNaturalist</a>
    </div>
    <div class="frog-cta"><button class="btn ${added ? "done" : "primary"}" id="life-toggle" data-label="${label}" type="button">${added
      ? `✓ On your Life List · ${new Date(added).toLocaleDateString([], { month: "short", day: "numeric" })}`
      : "Add to Life List"}</button></div>`;

  const g = $("gallery");
  g.addEventListener("scroll", () => {
    const i = Math.round(g.scrollLeft / g.clientWidth);
    if (pics[i]) { $("pic-credit").textContent = pics[i].attribution; $("pic-count").textContent = `${i + 1} of ${pics.length}`; }
  }, { passive: true });
  if ($("frog-hear")) $("frog-hear").addEventListener("click", () => $("frog-page").querySelector(".call").click());
}

async function showFrog(label) {
  if (!speciesBy[label]) return;
  if (!document.body.classList.contains("frog-open")) pageScroll = window.scrollY;
  document.body.classList.add("frog-open");
  $("frog-page").hidden = false;
  $("frog-bar").hidden = false;
  window.scrollTo(0, 0);
  const s = speciesBy[label];
  const cached = store.get(`taxon:${s.inat_taxon}`, null);
  renderFrog(label, cached);
  const d = await frogDetails(s);
  if (d && d !== cached && currentFrog() === label) renderFrog(label, d);
}

function hideFrog() {
  frogFromApp = false;
  stopCall();
  document.body.classList.remove("frog-open", "frog-scrolled");
  $("frog-page").hidden = true;
  $("frog-bar").hidden = true;
  $("frog-page").innerHTML = "";
  window.scrollTo(0, pageScroll);
}

const currentFrog = () => (location.hash.startsWith("#frog=") ? decodeURIComponent(location.hash.slice(6)) : null);

function route() {
  const l = currentFrog();
  if (l) showFrog(l); else if (document.body.classList.contains("frog-open")) hideFrog();
}

let frogFromApp = false; // opened by a tap here (so Back can go back), not from a shared link

function openSpecies(label) {
  if (currentFrog() === label) return;
  frogFromApp = true;
  location.hash = `frog=${encodeURIComponent(label)}`; // a history entry, so the phone's back gesture works too
}

function closeFrog() {
  if (frogFromApp) history.back();
  else { history.replaceState(null, "", location.pathname + location.search); route(); }
}

function stopCall() {
  if (frogPlayer) { frogPlayer.pause(); frogPlayer = null; }
  document.querySelectorAll(".call.playing").forEach((b) => b.classList.remove("playing"));
}

function playCall(btn) {
  const was = btn.classList.contains("playing");
  stopCall();
  if (was) return;
  const a = frogPlayer = new Audio(btn.dataset.src);
  btn.classList.add("playing");
  a.addEventListener("ended", () => { if (frogPlayer === a) stopCall(); });
  a.play().catch(() => { if (frogPlayer === a) stopCall(); });
}

async function shareFrog() {
  const s = speciesBy[currentFrog()];
  const url = location.href;
  try {
    if (navigator.share) await navigator.share({ title: `${s.name} · Frog ID`, url });
    else { await navigator.clipboard.writeText(url); alert("Link copied."); }
  } catch {}
}

function toggleLife(label) {
  if (life[label]) {
    if (!confirm(`Remove ${nameOf(label)} from your Life List?`)) return;
    delete life[label];
  } else life[label] = new Date().toISOString();
  store.set("life", life);
  renderLife();
  const b = $("life-toggle");
  const added = life[label];
  b.className = `btn ${added ? "done" : "primary"}`;
  b.textContent = added ? `✓ On your Life List · ${new Date(added).toLocaleDateString([], { month: "short", day: "numeric" })}` : "Add to Life List";
}

// ---------- results ----------

const heard = new Map(); // label -> {best, count, last}
let callingNow = new Set();

// a = answer scores, p = class scores (for the kind of noise), label = a recording's name (not live).
function showNow(a, p, label) {
  const frogs = calling(a);
  const meter = $("now-meter");
  const list = $("now-list");
  list.innerHTML = "";
  if (frogs.length > 1) {
    $("now-title").textContent = `${frogs.length} frogs ${label ? "heard" : "calling"}`;
    $("now-sub").textContent = label ? `in the ${label}` : "";
    meter.hidden = true;
    list.innerHTML = frogs.map(([j, v]) => `<li><span>${nameOf(answers[j])}</span><span class="muted">${Math.round(v * 100)}%</span>
      <div class="meter"><div style="width:${Math.round(v * 100)}%"></div></div></li>`).join("");
  } else if (frogs.length === 1) {
    const [j, v] = frogs[0];
    $("now-title").textContent = nameOf(answers[j]);
    $("now-sub").textContent = `${Math.round(v * 100)}% sure${label ? ` · ${label}` : ""}`;
    meter.hidden = false;
    meter.className = "meter";
    meter.firstElementChild.style.width = `${Math.round(v * 100)}%`;
  } else {
    let best = 0;
    for (let j = 1; j < answers.length - 1; j++) if (a[j] > a[best]) best = j;
    const nf = a[answers.length - 1];
    meter.hidden = false;
    meter.className = "meter quiet";
    if (nf >= a[best]) {
      $("now-title").textContent = "No frog";
      $("now-sub").textContent = `Sounds like: ${noiseKind(p).toLowerCase()} (${Math.round(nf * 100)}%)`;
      meter.firstElementChild.style.width = `${Math.round(nf * 100)}%`;
    } else {
      $("now-title").textContent = "Not sure";
      $("now-sub").textContent = `Maybe ${nameOf(answers[best])} (${Math.round(a[best] * 100)}%)`;
      meter.firstElementChild.style.width = `${Math.round(a[best] * 100)}%`;
    }
  }
  if (label) return; // a recording, not live listening
  callingNow = new Set(frogs.map(([j]) => answers[j]));
  for (const [j, v] of frogs) {
    const h = heard.get(answers[j]) || { best: 0, count: 0 };
    h.best = Math.max(h.best, v);
    h.count += 1;
    h.last = new Date();
    heard.set(answers[j], h);
  }
  renderHeard();
}

function renderHeard() {
  $("heard-h").hidden = heard.size === 0;
  $("heard").innerHTML = [...heard.entries()]
    .sort((x, y) => (callingNow.has(y[0]) - callingNow.has(x[0])) || (y[1].last - x[1].last))
    .map(([l, h]) => speciesRow(speciesBy[l] || { label: l, name: nameOf(l), scientific: "" },
      `<span class="sp-meta">${h.count} × · best ${Math.round(h.best * 100)}% · ${h.last.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}</span>${
      callingNow.has(l) ? `<span class="pill live">calling now</span>` : ""}`).replace("<li>", `<li class="${callingNow.has(l) ? "flash" : ""}">`))
    .join("");
  updateSessionRow();
}

function updateSessionRow() {
  const row = $("session-open");
  row.classList.toggle("live", !!audio);
  $("session-open-text").textContent = audio
    ? "Listening… tap to return"
    : heard.size ? `Sound ID session · ${heard.size} frog${heard.size === 1 ? "" : "s"} heard` : "Sound ID session";
}

function showSession(on) {
  $("home").hidden = on;
  $("session").hidden = !on;
  window.scrollTo(0, 0);
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
  $("listen").classList.add("on");
  $("listen").setAttribute("aria-label", "Stop");
  $("listen").innerHTML = `<svg><use href="#i-stop"/></svg>`;
  updateSessionRow();
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
  $("listen").classList.remove("on");
  $("listen").setAttribute("aria-label", "Listen");
  $("listen").innerHTML = `<svg><use href="#i-mic"/></svg>`;
  updateSessionRow();
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
  const rows = [], mean = new Array(answers.length).fill(0), best = new Array(answers.length).fill(0);
  const windowsHeard = new Array(answers.length).fill(0);
  for (let s = 0; s + WIN / 2 <= x.length && rows.length < MAX_FILE_WINDOWS; s += WIN) {
    const w = new Float32Array(WIN);
    w.set(x.subarray(s, s + WIN));
    const p = classProbs(embed(w));
    const a = toAnswers(p, false); // a file's place and date are unknown, so no filter
    a.forEach((v, j) => { mean[j] += v; best[j] = Math.max(best[j], v); });
    calling(a).forEach(([j]) => { windowsHeard[j] += 1; });
    rows.push({ t: s / SR, a, p });
    await new Promise((r) => setTimeout(r, 0)); // keep the page responsive
  }
  if (!rows.length) { box.innerHTML = `<h2>${title}</h2><p>That recording is too short (needs 1.5 s or more).</p>`; return; }
  mean.forEach((v, j) => { mean[j] = v / rows.length; });
  // Every frog that passed the threshold in at least one window, most windows first.
  const nf = answers.length - 1;
  const found = [];
  for (let j = 0; j < nf; j++) if (windowsHeard[j] > 0) found.push(j);
  found.sort((i, j) => windowsHeard[j] - windowsHeard[i] || best[j] - best[i]);
  const order = answers.map((v, j) => j).sort((i, j) => mean[j] - mean[i]);
  let verdict, detail;
  if (found.length) {
    verdict = found.length === 1 ? nameOf(answers[found[0]]) : `${found.length} frogs`;
    detail = `<ul class="top3">${found.map((j) => `<li>${nameOf(answers[j])} <span class="muted">in ${windowsHeard[j]} of ${rows.length} windows · best ${Math.round(best[j] * 100)}%</span></li>`).join("")}</ul>`;
  } else {
    const guess = order.find((j) => j !== nf);
    verdict = order[0] === nf ? "No frog" : `Not sure: maybe ${nameOf(answers[guess])}`;
    detail = `<p class="muted small">No frog reached ${Math.round(threshold * 100)}% in any 3 seconds. Best guesses:</p>
      <ol class="top3">${order.slice(0, 3).map((j) => `<li>${nameOf(answers[j])} <span class="muted">${Math.round(mean[j] * 100)}% on average</span></li>`).join("")}</ol>`;
  }
  showNow(best, rows[0].p, "recording");
  box.innerHTML = `<h2>${title}</h2>
    <p class="verdict">${verdict}</p>
    ${detail}
    <p class="muted small">The location and season filter is only used when listening live.</p>
    <h3>Every 3 seconds</h3>
    <ul class="windows">${rows.map((r) => {
      const fr = calling(r.a);
      let nm;
      if (fr.length) nm = fr.map(([j, v]) => `${nameOf(answers[j])} <span class="muted">${Math.round(v * 100)}%</span>`).join(", ");
      else {
        let b = 0;
        for (let j = 1; j < r.a.length; j++) if (r.a[j] > r.a[b]) b = j;
        nm = answers[b] === NOT_FROG ? `no frog (${noiseKind(r.p).toLowerCase()})` : `not sure (maybe ${nameOf(answers[b])})`;
        nm += ` <span class="muted">${Math.round(r.a[b] * 100)}%</span>`;
      }
      return `<li><b>${r.t.toFixed(0)}s</b> ${nm}</li>`;
    }).join("")}</ul>`;
}

// ---------- wiring ----------

function showTab(name) {
  if (currentFrog()) { history.replaceState(null, "", location.pathname + location.search); hideFrog(); }
  document.querySelectorAll(".tabbar button").forEach((x) => x.classList.toggle("active", x.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${name}`));
  document.body.dataset.tab = name;
  window.scrollTo(0, 0);
}

async function toggleListen() {
  if (audio) return stopListening();
  showSession(true);
  try { await startListening(); } catch (e) {
    $("now-title").textContent = "Can't use the microphone";
    $("now-sub").textContent = e.name === "NotAllowedError"
      ? "Allow microphone access for this site in your browser settings, then try again."
      : String(e.message || e);
  }
}

function setupUI() {
  document.querySelectorAll(".tabbar button").forEach((b) => b.addEventListener("click", () => {
    // Tapping Identify again goes back to its home screen.
    if (b.dataset.tab === "identify" && document.body.dataset.tab === "identify") showSession(false);
    showTab(b.dataset.tab);
  }));
  document.body.addEventListener("click", (e) => {
    const sp = e.target.closest(".sp, .hero, .cell");
    if (sp && sp.dataset.label) openSpecies(sp.dataset.label);
    const lt = e.target.closest("#life-toggle");
    if (lt) toggleLife(lt.dataset.label);
  });
  document.body.addEventListener("click", (e) => {
    const c = e.target.closest(".call");
    if (c) playCall(c);
  });
  $("frog-back").addEventListener("click", closeFrog);
  $("frog-share").addEventListener("click", shareFrog);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && currentFrog()) closeFrog(); });
  window.addEventListener("hashchange", route);
  window.addEventListener("scroll", () => {
    document.body.classList.toggle("frog-scrolled", document.body.classList.contains("frog-open") && window.scrollY > window.innerWidth * .62);
  }, { passive: true });

  $("listen").addEventListener("click", toggleListen);
  $("listen-home").addEventListener("click", toggleListen);
  $("session-open").addEventListener("click", () => showSession(true));
  $("session-back").addEventListener("click", () => showSession(false));

  let sampleAudio = null;
  $("file").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (!f || !model) return;
    if (sampleAudio) { sampleAudio.pause(); sampleAudio = null; }
    showSession(true);
    await analyzeBuffer(await f.arrayBuffer(), f.name);
    e.target.value = "";
  });

  $("sample").addEventListener("click", async () => {
    const s = samples[Math.floor(Math.random() * samples.length)];
    // Start playing right away, inside the tap: iPhones only allow sound that a tap starts.
    if (sampleAudio) sampleAudio.pause();
    const player = sampleAudio = new Audio(`samples/${s.file}`);
    player.controls = true;
    player.className = "player";
    player.play().catch(() => {}); // if it's blocked, the controls below still work
    showSession(true);
    const r = await fetch(`samples/${s.file}`);
    await analyzeBuffer(await r.arrayBuffer(), `Sample: ${s.name}`);
    if (player !== sampleAudio) return; // another sample was tapped meanwhile
    $("file-result").querySelector("h2").after(player);
    $("file-result").insertAdjacentHTML("beforeend",
      `<p class="muted small">Recording: ${s.attribution} (<a href="${s.url}" target="_blank" rel="noopener">iNaturalist</a>).
      The model was trained on this recording, so it's an easy test: it shows the app works, not how accurate it is.</p>`);
  });

  $("search").addEventListener("input", renderExplore);
  $("use-filter").addEventListener("change", () => { useFilter = $("use-filter").checked; store.set("useFilter", useFilter); renderAll(); });
  $("show-sci").addEventListener("change", () => { showSci = $("show-sci").checked; store.set("showSci", showSci); renderAll(); renderHeard(); });
  $("locate").addEventListener("click", locate);
  $("find-locate").addEventListener("click", locate);
  $("explore-filter").addEventListener("click", () => showTab("settings"));
  $("filter-note").addEventListener("click", () => showTab("settings"));
  $("life-info").addEventListener("click", () => alert(
    "Your Life List is the frogs you've found. Open any frog and tap Add to Life List. It's saved on this phone only."));
  $("threshold").value = String(threshold);
  $("threshold").addEventListener("change", () => { threshold = Number($("threshold").value); store.set("threshold", threshold); });
  $("version").textContent = `Version ${VERSION}`;
}

// ---------- offline ----------

const standalone = window.matchMedia("(display-mode: standalone)").matches || navigator.standalone === true;

async function setupOffline() {
  $("install-steps").hidden = standalone;
  const status = $("offline-status");
  if (!("serviceWorker" in navigator)) { status.textContent = "This browser can't save the app for offline use."; return; }
  try {
    // A new version took over (sw.js VERSION changed): reload once to use it, unless listening.
    const hadOld = !!navigator.serviceWorker.controller;
    navigator.serviceWorker.addEventListener("controllerchange", () => {
      if (hadOld && !audio) location.reload();
    });
    await navigator.serviceWorker.register("sw.js");
    await navigator.serviceWorker.ready;
    const saved = await caches.match("model/birdnet_embed.tflite", { ignoreSearch: true });
    status.innerHTML = saved
      ? `<span class="ok">✓ Saved on this ${standalone ? "Home Screen app" : "browser"}.</span> Works with no signal.`
      : "Saving for offline use once the model has loaded…";
    status.classList.toggle("muted", !saved);
  } catch (e) {
    status.textContent = `Couldn't save it for offline use (${e.message || e}).`;
  }
}

setupUI();
loadAll().then(setupOffline, (e) => {
  console.error(e);
  $("load-note").textContent = navigator.onLine
    ? `Couldn't load the model: ${e.message || e}`
    : "You're offline and the model isn't saved on this phone yet. Open the app once with signal.";
  setupOffline();
});

// For testing from the console.
window.frogid = { analyzeBuffer, classProbs, embed, toAnswers, showNow, get answers() { return answers; } };
