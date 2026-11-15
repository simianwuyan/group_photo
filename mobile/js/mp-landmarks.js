// MediaPipe tasks-vision FaceLandmarker 封装（landmarks.py 的 JS 对应）。
// 478 点 = 468 基础网格 + 10 虹膜；头姿从 facialTransformationMatrixes 取。
// 索引常量与 gpmerge/landmarks.py 一致。

import { FaceLandmarker, FilesetResolver } from '../vendor/tasks-vision/vision_bundle.mjs';

export const IRIS_L = 473;
export const IRIS_R = 468;
export const ALIGN_ANCHORS = [33, 263, 133, 362, 1, 61, 291, 152];

export async function createLandmarker({ wasmBase, modelBuffer, numFaces = 30, runningMode = 'IMAGE' }) {
  const fileset = await FilesetResolver.forVisionTasks(wasmBase);
  const opts = (delegate) => ({
    baseOptions: { modelAssetBuffer: modelBuffer, delegate },
    runningMode,
    numFaces,
    outputFacialTransformationMatrixes: true,
  });
  let lm;
  try {
    lm = await FaceLandmarker.createFromOptions(fileset, opts('GPU'));
  } catch (e) {
    console.warn('[landmarks] GPU delegate 不可用，回退 CPU：', e);
    lm = await FaceLandmarker.createFromOptions(fileset, opts('CPU'));
  }
  return { detectFaces: (source, width, height, frameIdx) => detectFaces(lm, source, width, height, frameIdx) };
}

// 页面自取模型并校验 zip 完整性——不信任 mediapipe 内部 fetch 与 SW 缓存层。
// 校验三项：大小 ≥ 1MB、头部 PK 魔数、**尾部 EOCD 记录**（截断的 zip 只以 PK
// 开头而无中央目录，之前仅校验 PK 时截断模型会漏过并导致 mediapipe zip 错误）。
// 成功后把正确字节写回所有 gpmerge 缓存，自愈可能存在的坏条目。
export function looksLikeCompleteModel(buf) {
  if (buf.length < 1000000) return false;                      // 模型应约 3.7MB
  if (!(buf[0] === 0x50 && buf[1] === 0x4b)) return false;     // zip 头 PK\x03\x04
  const scanStart = Math.max(0, buf.length - 66000);           // EOCD 在尾部
  for (let i = buf.length - 22; i >= scanStart; i--) {
    if (buf[i] === 0x50 && buf[i + 1] === 0x4b && buf[i + 2] === 0x05 && buf[i + 3] === 0x06) {
      return true;                                             // 找到 EOCD PK\x05\x06
    }
  }
  return false;
}

export async function fetchValidatedModel(modelUrl, attempts = 5, onStatus = () => {}) {
  let lastErr = null;
  for (let i = 0; i < attempts; i++) {
    const url = i === 0 ? modelUrl : `${modelUrl}?r=${i}&t=${Date.now()}`;
    onStatus(i === 0 ? '下载模型…' : `重试下载模型（第 ${i + 1}/${attempts} 次）…`);
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const buf = new Uint8Array(await res.arrayBuffer());
      if (looksLikeCompleteModel(buf)) {
        onStatus(`模型校验通过（${(buf.length / 1048576).toFixed(1)}MB）`);
        // 自愈：把正确字节写回所有 gpmerge 缓存，修复可能的坏条目
        try {
          const keys = await caches.keys();
          for (const k of keys) {
            if (!k.startsWith('gpmerge-mobile-')) continue;
            const c = await caches.open(k);
            await c.put(modelUrl, new Response(buf, {
              headers: { 'content-type': 'application/octet-stream' },
            }));
          }
        } catch { /* 自愈失败不影响主流程 */ }
        return buf;
      }
      lastErr = new Error(`第 ${i + 1} 次取到无效内容（${buf.length} 字节，模型不完整）`);
      console.warn('[landmarks]', lastErr.message);
    } catch (e) {
      lastErr = e;
      console.warn('[landmarks] 模型下载失败:', e.message);
    }
    if (i < attempts - 1) await new Promise((r) => setTimeout(r, 800 * (i + 1)));
  }
  throw lastErr || new Error('模型校验失败');
}

export function makeFace(pts, frameIdx) {
  // pts: Array<[x, y]> 像素坐标（478 点）
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  let cx = 0, cy = 0;
  for (const [x, y] of pts) {
    if (x < x0) x0 = x; if (x > x1) x1 = x;
    if (y < y0) y0 = y; if (y > y1) y1 = y;
    cx += x; cy += y;
  }
  const n = pts.length;
  return {
    pts,
    frameIdx,
    personId: null,
    headYaw: null,
    headPitch: null,
    box: { x: Math.trunc(x0), y: Math.trunc(y0),
           w: Math.trunc(x1 - x0), h: Math.trunc(y1 - y0) },
    center: [cx / n, cy / n],
    faceWidth: Math.trunc(x1 - x0),
    ipd: Math.hypot(pts[33][0] - pts[263][0], pts[33][1] - pts[263][1]),  // 外眼角距离
  };
}

// 在一张图（canvas/ImageBitmap，含宽高）上检测所有人脸。
export function detectFaces(lm, source, width, height, frameIdx) {
  const result = lm.detect(source);
  const faces = result.faceLandmarks.map((lms, i) => {
    const pts = lms.map((p) => [p.x * width, p.y * height]);
    const face = makeFace(pts, frameIdx);
    // 头部姿态：变换矩阵旋转部分第三列 = 人脸法向（相机坐标系），
    // 正对镜头时 yaw/pitch≈0。Matrix.data 为列主序（rows×columns）。
    const m = result.facialTransformationMatrixes && result.facialTransformationMatrixes[i];
    if (m && m.data && m.data.length >= 12) {
      const d = m.data;
      const col = m.rows === 4 && m.columns === 4
        ? [d[8], d[9], d[10]]                      // column-major：第 3 列
        : [d[2], d[6], d[10]];                     // 兜底：row-major 第 3 列
      const len = Math.hypot(col[0], col[1], col[2]) + 1e-6;
      const nx = col[0] / len, ny = col[1] / len, nz = col[2] / len;
      face.headYaw = (Math.atan2(nx, nz) * 180) / Math.PI;
      face.headPitch = (Math.asin(Math.max(-1, Math.min(1, ny))) * 180) / Math.PI;
    }
    return face;
  });
  faces.sort((a, b) => a.center[0] - b.center[0]);  // 按 x 排序（对齐 Python）
  return faces;
}
