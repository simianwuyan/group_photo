// PWA 预缓存：全部算法资源离线可用（首次联网后即可飞行模式使用）。
// 注意：运行时 fetch 不做缓存写回——慢网中断会产生"头完整 body 截断"的坏条目
// （曾导致模型 zip 解析失败且刷新无效）；install 的 addAll 具有原子完整性。
const CACHE = 'gpmerge-mobile-v4';
const ASSETS = [
  './',
  './index.html',
  './test.html',
  './manifest.json',
  './js/main.js',
  './js/ocv.js',
  './js/scoring.js',
  './js/tracking.js',
  './js/umeyama.js',
  './js/mp-landmarks.js',
  './js/compositing.js',
  './js/pipeline.js',
  './vendor/opencv.js',
  './vendor/tasks-vision/vision_bundle.mjs',
  './vendor/tasks-vision/wasm/vision_wasm_internal.js',
  './vendor/tasks-vision/wasm/vision_wasm_internal.wasm',
  './vendor/tasks-vision/wasm/vision_wasm_module_internal.js',
  './vendor/tasks-vision/wasm/vision_wasm_module_internal.wasm',
  './models/face_landmarker.task',
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((keys) =>
    Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
  ).then(() => self.clients.claim()));
});

self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;
  // 预缓存命中直接用；未预缓存的资源直连网络（不写回缓存）
  e.respondWith(caches.match(e.request).then((hit) => hit || fetch(e.request)));
});
