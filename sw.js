// Service worker: keeps the whole app on the phone so it opens with no signal.
// Bump VERSION whenever any file below changes, or phones keep the old copy.
const VERSION = "0.3";
const SHELL = `frog-id-shell-${VERSION}`;
const RUNTIME = "frog-id-runtime"; // sample clips, cached the first time they play
const FILES = [
  "./", "index.html", "app.js", "style.css", "manifest.json",
  "icon.svg", "icon-192.png", "icon-512.png", "apple-touch-icon.png",
  "model/frog_classifier.json", "model/filter.json", "model/species.json",
  "samples/samples.json",
  "vendor/tf-core.min.js", "vendor/tf-backend-cpu.min.js", "vendor/tf-tflite.min.js",
  "vendor/tflite_web_api_cc.js", "vendor/tflite_web_api_cc.wasm",
  "vendor/tflite_web_api_cc_simd.js", "vendor/tflite_web_api_cc_simd.wasm",
];
// The 25 MB BirdNET model is saved by app.js in its own cache ("frog-id-model-v1"),
// which survives app updates; it's served from there like everything else.

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(SHELL).then((c) => c.addAll(FILES.map((f) => new Request(f, { cache: "reload" })))));
  self.skipWaiting();
});

self.addEventListener("activate", (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k.startsWith("frog-id-shell-") && k !== SHELL) await caches.delete(k);
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET" || new URL(req.url).origin !== location.origin) return;
  const range = req.headers.get("range");
  e.respondWith((async () => {
    let res = await caches.match(req.url, { ignoreSearch: true });
    if (!res) {
      res = await fetch(range ? req.url : req); // whole file, so it can be cached
      if (res.ok && new URL(req.url).pathname.includes("/samples/")) {
        const c = await caches.open(RUNTIME);
        await c.put(req.url, res.clone());
      }
    }
    return range && res.ok ? partial(res, range) : res;
  })());
});

// Safari plays audio with byte-range requests and needs a 206 answer.
async function partial(res, range) {
  const buf = await res.arrayBuffer();
  const m = /bytes=(\d*)-(\d*)/.exec(range) || [];
  let start = m[1] ? Number(m[1]) : 0;
  let end = m[2] ? Number(m[2]) : buf.byteLength - 1;
  if (!m[1] && m[2]) { start = Math.max(0, buf.byteLength - Number(m[2])); end = buf.byteLength - 1; }
  end = Math.min(end, buf.byteLength - 1);
  return new Response(buf.slice(start, end + 1), {
    status: 206,
    headers: {
      "Content-Type": res.headers.get("Content-Type") || "application/octet-stream",
      "Content-Range": `bytes ${start}-${end}/${buf.byteLength}`,
      "Content-Length": String(end - start + 1),
      "Accept-Ranges": "bytes",
    },
  });
}
