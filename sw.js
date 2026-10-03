// App-shell cache (network-first for our files so updates arrive; cache-first for the CDN detector/model).
const V = "vc-v4";
const SHELL = ["./", "index.html", "css/app.css", "js/app.js", "js/gpu.js", "js/optics.js", "js/content.js", "js/tracker.js", "js/methods.js", "js/hdr.js", "js/cpu.js", "manifest.webmanifest", "icons/icon-192.png", "icons/icon-512.png"];
self.addEventListener("install", (e) => { e.waitUntil(caches.open(V).then(c => c.addAll(SHELL))); self.skipWaiting(); });
self.addEventListener("activate", (e) => { e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== V).map(k => caches.delete(k))))); self.clients.claim(); });
self.addEventListener("fetch", (e) => {
  const u = new URL(e.request.url);
  if (e.request.method !== "GET") return;
  const cdn = u.hostname.includes("jsdelivr") || u.hostname.includes("storage.googleapis.com");
  if (cdn) { e.respondWith(caches.open(V).then(async c => (await c.match(e.request)) || fetch(e.request).then(r => { if (r.ok) c.put(e.request, r.clone()); return r; }))); return; }
  if (u.origin === location.origin) e.respondWith(fetch(e.request).then(r => { const cl = r.clone(); caches.open(V).then(c => c.put(e.request, cl)); return r; }).catch(() => caches.match(e.request)));
});
