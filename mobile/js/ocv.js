// opencv.js 加载器：fetch 带进度 → blob URL 注入 script → 解析 thenable。
// @techstark 5.0 构建为 UMD + MODULARIZE：window.cv 是 thenable 壳，
// 真正带符号（Mat/warpAffine…）的模块实例在 cv.then() 的 resolve 参数里，
// 就绪后挂回 window.cv 供全站代码使用。wasm 以 base64 内嵌，无外部依赖。

const WAIT_MS = 180000;   // 弱机 WASM 编译可达分钟级

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
        (async () => {
          while (!window.cv && Date.now() - t0 < WAIT_MS) await new Promise((r) => setTimeout(r, 50));
          if (!window.cv) throw new Error('window.cv 未出现（脚本执行异常）');
          // MODULARIZE thenable：await 拿到带符号的模块实例
          const mod = typeof window.cv.then === 'function'
            ? await new Promise((r) => window.cv.then(r)) : window.cv;
          window.cv = (mod && mod.Mat) ? mod : window.cv;
          if (!window.cv.Mat) throw new Error('OpenCV.js 初始化异常：符号未挂载');
          resolve(window.cv);
        })().catch(reject);
      };
      el.onerror = () => reject(new Error('opencv.js 脚本注入失败'));
      document.head.appendChild(el);
    }).catch(reject);
  });
}
