// 逐人逐帧质量评分（scoring.py 逐字移植）。
// 原则：能用几何量（关键点）算的就不上模型——快、可解释、端侧零成本。
// 睁眼分数依赖"同一个人跨帧归一化"（pipeline 里做），这里提供原始量。

export const EAR_BLINK_ABS = 0.14;   // 低于该值基本可判定闭眼
export const EAR_RATIO_OPEN = 0.72;  // 相对阈值：ear / 个人全场最大 EAR

// EAR 六点索引（与 gpmerge/landmarks.py 一致）
const LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380];

export const TOTAL_WEIGHTS = [0.45, 0.20, 0.10, 0.15, 0.10]; // 睁眼权重最高

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

// 眼睛纵横比：双眼均值，越小越闭。
// one_eye: v1=|p1-p5|, v2=|p2-p4|, h=|p0-p3|, ear=(v1+v2)/(2h+1e-6)
export function earValue(face) {
  const oneEye = (idx6) => {
    const p = idx6.map((i) => face.pts[i]);
    const v1 = dist(p[1], p[5]);
    const v2 = dist(p[2], p[4]);
    const h = dist(p[0], p[3]);
    return (v1 + v2) / (2 * h + 1e-6);
  };
  return (oneEye(LEFT_EYE_EAR) + oneEye(RIGHT_EYE_EAR)) / 2;
}

// 睁眼分数 ∈ [0,1]：绝对阈值 + 相对阈值双保险。
export function eyesOpenScore(ear, personMaxEar) {
  const denom = Math.max(personMaxEar, EAR_BLINK_ABS);
  const ratio = ear / denom;
  if (ratio >= EAR_RATIO_OPEN) {
    return clamp(0.5 + 0.5 * (ratio - EAR_RATIO_OPEN) / (1 - EAR_RATIO_OPEN + 1e-6), 0.5, 1.0);
  }
  return clamp(ratio / (2 * EAR_RATIO_OPEN), 0.0, 0.5);
}

// 微笑分数：嘴角间距相对瞳距 + 嘴角相对唇中心上提量。
export function smileScore(face) {
  const ipd = face.ipd;
  const mouthW = dist(face.pts[61], face.pts[291]) / ipd;
  const cornerY = (face.pts[61][1] + face.pts[291][1]) / 2;
  const lipY = (face.pts[13][1] + face.pts[14][1]) / 2;
  const lift = (lipY - cornerY) / ipd;  // 嘴角高于唇中心为正
  const s = 0.5 * clamp((mouthW - 0.28) / 0.25, 0, 1)
          + 0.5 * clamp((lift + 0.02) / 0.12, 0, 1);
  return clamp(s, 0, 1);
}

// 视线/朝向分数：虹膜居中程度 + 头部正对程度。
export function gazeScore(face) {
  const irisCentering = (iris, outer, inner) => {
    const eyeMid = [(face.pts[outer][0] + face.pts[inner][0]) / 2,
                    (face.pts[outer][1] + face.pts[inner][1]) / 2];
    const eyeW = dist(face.pts[outer], face.pts[inner]) + 1e-6;
    const off = dist(face.pts[iris], eyeMid) / eyeW;
    return clamp(1.0 - off * 1.6, 0, 1);
  };
  const c = (irisCentering(473, 33, 133) + irisCentering(468, 362, 263)) / 2;
  if (face.headYaw !== null && face.headYaw !== undefined) {
    // 偏航/俯仰每偏 15° 扣一档
    const ang = Math.hypot(face.headYaw, face.headPitch);
    const head = Math.exp(-((ang / 18.0) ** 2));
    return 0.6 * c + 0.4 * head;
  }
  return c;
}

// 清晰度：人脸区域 Laplacian 方差（CV_64F，BORDER_REFLECT_101），映射到 [0,1]。
// grayFace: {data: Uint8ClampedArray|Uint8Array, w, h}
export function sharpnessScore(grayFace) {
  const { data, w, h } = grayFace;
  const idx101 = (i, n) => { if (i < 0) i = -i; if (i >= n) i = 2 * (n - 1) - i; return i; };
  const sq = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const c = idx101(y, h) * w + idx101(x, w);       // 中心（含边界的 reflect101 索引）
      const up = idx101(y - 1, h) * w + idx101(x, w);
      const dn = idx101(y + 1, h) * w + idx101(x, w);
      const lf = idx101(y, h) * w + idx101(x - 1, w);
      const rt = idx101(y, h) * w + idx101(x + 1, w);
      sq[y * w + x] = data[up] + data[dn] + data[lf] + data[rt] - 4 * data[c];
    }
  }
  let s1 = 0, s2 = 0;
  for (let i = 0; i < sq.length; i++) { s1 += sq[i]; s2 += sq[i] * sq[i]; }
  const n = sq.length;
  const v = s2 / n - (s1 / n) ** 2;   // 总体方差（对齐 numpy .var()）
  return clamp(v / 250.0, 0, 1);
}

// 亮度：均值过暗/过曝都扣分，中间调得 1。
export function brightnessScore(grayFace) {
  const { data } = grayFace;
  let s = 0;
  for (let i = 0; i < data.length; i++) s += data[i];
  const m = s / data.length / 255.0;
  return Math.exp(-(((m - 0.55) / 0.28) ** 2));
}

// 加权总分。睁眼权重最高——合影废片第一杀手。
export function totalScore(eyes, sharp, bright, smile, gaze) {
  const v = [eyes, sharp, bright, smile, gaze];
  let t = 0;
  for (let i = 0; i < v.length; i++) t += v[i] * TOTAL_WEIGHTS[i];
  return t;
}
