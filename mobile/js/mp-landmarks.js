// MediaPipe tasks-vision FaceLandmarker 封装（landmarks.py 的 JS 对应）。
// 478 点 = 468 基础网格 + 10 虹膜；头姿从 facialTransformationMatrixes 取。
// 索引常量与 gpmerge/landmarks.py 一致。

import { FaceLandmarker, FilesetResolver } from '../vendor/tasks-vision/vision_bundle.mjs';

export const IRIS_L = 473;
export const IRIS_R = 468;
export const ALIGN_ANCHORS = [33, 263, 133, 362, 1, 61, 291, 152];

export async function createLandmarker({ wasmBase, modelUrl, numFaces = 30, runningMode = 'IMAGE' }) {
  const fileset = await FilesetResolver.forVisionTasks(wasmBase);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: modelUrl, delegate },
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
