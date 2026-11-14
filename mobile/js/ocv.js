// opencv.js 加载器：fetch 带进度 → blob URL 注入 script → 轮询等待 window.cv。
// （13MB 单文件，wasm 以 base64 内嵌；PWA 缓存后二次访问秒开。）

export function loadOpenCv({ url = 'vendor/opencv.js', onProgress = () => {} } = {}) {
  return new Promise((resolve, reject) => {
    if (window.cv && window.cv.Mat) return resolve(window.cv);
    fetch(url).then(async (res) => {
      if (!res.ok) throw new Error(`opencv.js 下载失败: HTTP ${res.status}`);
      const total = +(res.headers.get('content-length') || 0);
      if (!res.body) return res.blob();
      const reader = res.body.getReader();
      const chunks = [];
      let loaded = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value); loaded += value.length;
        onProgress(total ? loaded / total : 0, loaded, total);
      }
      onProgress(1, loaded, total);
      return new Blob(chunks, { type: 'application/javascript' });
    }).then((blob) => {
      const el = document.createElement('script');
      el.src = URL.createObjectURL(blob);
      el.onload = () => {
        const t0 = Date.now();
        (function waitCv() {
          if (window.cv && window.cv.Mat) return resolve(window.cv);
          if (Date.now() - t0 > 30000) return reject(new Error('OpenCV.js 初始化超时'));
          setTimeout(waitCv, 50);
        })();
      };
      el.onerror = () => reject(new Error('opencv.js 脚本注入失败'));
      document.head.appendChild(el);
    }).catch(reject);
  });
}
