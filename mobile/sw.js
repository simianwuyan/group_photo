// PWA 预缓存：全部算法资源离线可用（首次联网后即可飞行模式使用）。
// 两个防御（均为线上踩坑后加固）：
// 1. 运行时 fetch 不做缓存写回——慢网中断会产生"头完整 body 截断"的坏条目；
// 2. install 逐个缓存并对模型校验 PK 魔数——坏响应不进缓存（页面侧
//    fetchValidatedModel 会自取并自愈写回正确字节）。
const CACHE = 'gpmerge-mobile-v5';
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
  e.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    for (const url of ASSETS) {
      try {
        const res = await fetch(url);
        if (!res.ok) continue;
        // 模型类资源校验 zip 魔数：坏响应不进缓存
        if (url.endsWith('.task')) {
          const buf = new Uint8Array(await res.clone().arrayBuffer());
          if (!(buf[0] === 0x50 && buf[1] === 0x4b)) continue;
        }
        await cache.put(url, res);
      } catch { /* 单项失败跳过，页面侧自取兜底 */ }
    }
    await self.skipWaiting();
  })());
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
