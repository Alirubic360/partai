// App-shell cache only. API calls are never cached.
const C = "partai-v1", SHELL = ["/", "/manifest.json", "/icon.svg"];
self.addEventListener("install", e => e.waitUntil(caches.open(C).then(c => c.addAll(SHELL))));
self.addEventListener("fetch", e => {
  const u = new URL(e.request.url);
  if (u.pathname.startsWith("/api/") || e.request.method !== "GET") return;
  e.respondWith(fetch(e.request).catch(() => caches.match(e.request)));
});
