// 手机端主逻辑：资源预加载 → 摄像头连拍 → 本地合并 → 结果展示。
import { loadOpenCv } from './ocv.js';
import { createLandmarker, fetchValidatedModel } from './mp-landmarks.js';
import { mergeBurst } from './pipeline.js';
import { matToCanvas } from './compositing.js';

const $ = (id) => document.getElementById(id);
const barEl = $('bar').firstElementChild;
const msgEl = $('msg');
const resultEl = $('result');
const tableEl = $('table');
const video = $('video');
const startBtn = $('startBtn');
const closeBtn = $('closeBtn');
const flipBtn = $('flipBtn');
const burstBtn = $('burstBtn');

let stream = null, facing = 'environment';
let lm = null;
const say = (t) => { msgEl.textContent = t || ''; };
const setBar = (f) => { barEl.style.width = `${Math.round(f * 100)}%`; };

// ---- PWA ----
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('sw.js').catch(() => {});
}

// ---- 资源预加载（OpenCV 与模型并行下载；慢网时给出明确提示）----
async function preload() {
  burstBtn.disabled = true;
  say('加载算法资源（首次约 15MB，之后离线可用）…');
  let lastProg = Date.now();
  const slowTimer = setInterval(() => {
    if (Date.now() - lastProg > 12000 && !lm) {
      say('网络较慢，资源下载停滞…建议：① 开启代理后刷新；或 ② 使用国内镜像站点（魔搭）');
    }
  }, 3000);
  try {
    const cvReady = loadOpenCv({
      onProgress: (f) => { setBar(f * 0.6); lastProg = Date.now(); },
    });
    const lmReady = (async () => {
      // 先自取模型校验+自愈缓存（保证缓存条目完好），再用 path 加载
      await fetchValidatedModel('models/face_landmarker.task',
        5, (s) => { say(s); lastProg = Date.now(); });
      setBar(0.8);
      lm = await createLandmarker({
        wasmBase: 'vendor/tasks-vision/wasm',
        modelUrl: 'models/face_landmarker.task',
      });
    })();
    await Promise.all([cvReady, lmReady]);
    setBar(1);
    say('');
  } finally {
    clearInterval(slowTimer);
  }
}
preload().catch((e) => { setBar(0); say('资源加载失败：' + e.message + '（刷新重试）'); });

// ---- 摄像头 ----
async function openCamera() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    flipBtn.disabled = false;
    closeBtn.disabled = false;
    burstBtn.disabled = !lm;
    say('');
  } catch (e) {
    say('摄像头打开失败：' + e.message + '（检查权限是否允许、是否被占用）');
  }
}
startBtn.onclick = () => { startBtn.textContent = '重新开启'; openCamera(); };
flipBtn.onclick = () => { facing = facing === 'environment' ? 'user' : 'environment'; openCamera(); };

// 关闭摄像头：停流并复位按钮
closeBtn.onclick = () => {
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  video.srcObject = null;
  flipBtn.disabled = true;
  burstBtn.disabled = true;
  startBtn.textContent = '开启摄像头';
  say('');
};

// 紧急重置：注销 SW + 清空本站全部缓存并重载（修复历史坏缓存条目）
$('resetBtn').onclick = async () => {
  say('正在清理缓存并刷新…');
  try {
    const regs = await navigator.serviceWorker.getRegistrations();
    regs.forEach((r) => r.unregister());
    const keys = await caches.keys();
    await Promise.all(keys.map((k) => caches.delete(k)));
  } catch { /* 忽略清理错误，直接刷新 */ }
  location.reload();
};

// ---- 连拍 ----
async function captureFrames(n, gapMs) {
  const out = [];
  const cnv = new OffscreenCanvas(video.videoWidth, video.videoHeight);
  const ctx = cnv.getContext('2d');
  for (let i = 0; i < n; i++) {
    const t0 = performance.now();
    ctx.drawImage(video, 0, 0);
    out.push(await createImageBitmap(cnv));
    say(`已抓取 ${i + 1}/${n} 帧`);
    setBar(((i + 1) / n) * 0.2);
    const wait = gapMs - (performance.now() - t0);
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
  }
  return out;
}

// ---- 连拍 + 合并 ----
burstBtn.onclick = async () => {
  if (!stream || !video.videoWidth) { say('请先开启摄像头'); return; }
  if (!lm) { say('算法资源还在加载，请稍候'); return; }
  burstBtn.disabled = true; startBtn.disabled = true; flipBtn.disabled = true;
  resultEl.innerHTML = ''; tableEl.style.display = 'none';
  try {
    const n = Math.max(6, Math.min(16, +$('nFrames').value || 12));
    const bitmaps = await captureFrames(n, 130);
    say('本地合并中（无需联网）…');
    const images = bitmaps.map((b, i) => ({ source: b, name: `frame_${String(i).padStart(3, '0')}.jpg` }));
    const res = await mergeBurst(images, lm, undefined, (stage, f) => {
      setBar(0.2 + f * 0.8);
      say(stage === 'detect' ? `人脸检测 ${(f * 100).toFixed(0)}%`
        : stage === 'score' ? `逐人评分 ${(f * 100).toFixed(0)}%`
        : stage === 'merge' ? '换脸合成中…' : '');
    });

    say(`基准底片第 ${res.baseFrameIdx + 1} 帧，替换 `
        + `${res.rows.filter((r) => r.action !== 'kept').length} 人 / 共 ${res.rows.length} 人`);
    const mergedURL = (() => {
      const c = document.createElement('canvas');
      matToCanvas(res.merged, c);
      return c.toDataURL('image/jpeg', 0.92);
    })();
    const annotURL = (() => {
      const c = document.createElement('canvas');
      matToCanvas(res.annotated, c);
      return c.toDataURL('image/jpeg', 0.92);
    })();
    const mk = (src, cap) => {
      const fig = document.createElement('figure');
      fig.style.margin = '0';
      fig.innerHTML = `<figcaption style="font-size:13px;color:#93a4bb;margin:8px 0 4px">${cap}</figcaption>`;
      const img = new Image(); img.src = src; img.style.width = '100%';
      fig.appendChild(img);
      resultEl.appendChild(fig);
    };
    mk(mergedURL, '合并成片（长按图片保存）');
    mk(annotURL, '标注图（绿=保留 橙=替换）');

    tableEl.style.display = '';
    tableEl.innerHTML = '<tr><th>人</th><th>动作</th><th>基准分</th><th>来源分</th><th>EAR基准</th><th>EAR来源</th></tr>';
    for (const r of res.rows) {
      const tr = document.createElement('tr');
      const act = r.action === 'kept' ? '保留'
        : r.action === 'swapped' ? `替换 ← 第${r.srcFrame + 1}帧`
        : `补回 ← 第${r.srcFrame + 1}帧`;
      for (const c of [`P${r.personId}`, act,
                       r.scoreBase.toFixed(3), r.scoreSrc.toFixed(3),
                       r.earBase.toFixed(3), r.earSrc.toFixed(3)]) {
        const td = document.createElement('td'); td.textContent = c; tr.appendChild(td);
      }
      tableEl.appendChild(tr);
    }
    for (const b of bitmaps) b.close?.();
  } catch (e) {
    say('出错了：' + e.message);
    console.error(e);
  } finally {
    burstBtn.disabled = false; startBtn.disabled = false; flipBtn.disabled = false;
  }
};
