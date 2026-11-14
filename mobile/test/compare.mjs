// node 对照验证：JS 移植模块 vs Python 导出的参考数据（ref.json）。
// 用法：node test/compare.mjs
// 覆盖：scoring 全指标 / tracking 关联 / umeyama 相似变换 / 合并决策流。

import { readFileSync } from 'node:fs';
import { earValue, eyesOpenScore, smileScore, gazeScore,
         sharpnessScore, brightnessScore, totalScore } from '../js/scoring.js';
import { associate } from '../js/tracking.js';
import { estimateSimilarityTransform } from '../js/umeyama.js';

// 用法：node test/compare.mjs [ref.json 路径]
const refPath = process.argv[2] || new URL('./ref.json', import.meta.url).href;
const ref = JSON.parse(readFileSync(refPath, 'utf-8'));
console.log(`对照 ${refPath}（基准帧 ${ref.base_idx}，rows=${ref.rows.length}）`);

let pass = 0, fail = 0;
function check(name, ok, detail = '') {
  if (ok) { pass++; console.log(`  OK   ${name}`); }
  else { fail++; console.log(`  FAIL ${name} ${detail}`); }
}

// face 对象（pts 用 ref 原值重建，ipd/center/faceWidth 按 landmarks.py 语义）
function rebuildFace(fj) {
  const pts = fj.pts;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  let cx = 0, cy = 0;
  for (const [x, y] of pts) {
    if (x < x0) x0 = x; if (x > x1) x1 = x;
    if (y < y0) y0 = y; if (y > y1) y1 = y;
    cx += x; cy += y;
  }
  const n = pts.length;
  return { pts, frameIdx: -1, personId: null,
    headYaw: fj.headYaw, headPitch: fj.headPitch,
    box: { x: Math.trunc(x0), y: Math.trunc(y0), w: Math.trunc(x1 - x0), h: Math.trunc(y1 - y0) },
    center: [cx / n, cy / n],
    faceWidth: Math.trunc(x1 - x0),
    ipd: Math.hypot(pts[33][0] - pts[263][0], pts[33][1] - pts[263][1]) };
}

const near = (a, b, tol = 1e-9) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));
// Python 侧关键点以 float32 参与计算（mediapipe 输出 float32），自带 ~1e-7 噪声；
// JS 用 float64 更精确，故对照容差定为 5e-6（远小于任何决策阈值）。
const SCORE_TOL = 5e-6;

// ---- 1. scoring：逐人逐帧五维指标 + 总分 ----
console.log('[1] scoring');
const jsScores = new Map();
let scoreFails = 0;
for (const [pid, perRef] of Object.entries(ref.raw)) {
  const perScore = new Map();
  for (const [fiS, mRef] of Object.entries(perRef)) {
    const fi = +fiS;
    const fj = ref.framesFaces[fi].find((f) => f.personId === +pid);
    const face = rebuildFace(fj);
    const ear = earValue(face);
    const smile = smileScore(face);
    const gaze = gazeScore(face);
    const roi = ref.rois[`${pid}_${fi}`];
    const sharp = sharpnessScore(roi);
    const bright = brightnessScore(roi);
    const eyes = eyesOpenScore(ear, Math.max(...Object.values(perRef).map((m) => m.ear)));
    const total = totalScore(eyes, sharp, bright, smile, gaze);
    const okEar = near(ear, mRef.ear, SCORE_TOL);
    const okSmile = near(smile, mRef.smile, SCORE_TOL);
    const okGaze = near(gaze, mRef.gaze, SCORE_TOL);
    const okSharp = near(sharp, mRef.sharp, SCORE_TOL);
    const okBright = near(bright, mRef.bright, SCORE_TOL);
    const okTotal = near(total, ref.scores[pid][fiS], SCORE_TOL);
    if (!(okEar && okSmile && okGaze && okSharp && okBright && okTotal)) {
      scoreFails++;
      check(`P${pid} f${fi} 指标`, false,
        `ear ${ear.toFixed(6)}/${mRef.ear.toFixed(6)} smile ${smile.toFixed(6)}/${mRef.smile.toFixed(6)} ` +
        `gaze ${gaze.toFixed(6)}/${mRef.gaze.toFixed(6)} sharp ${sharp.toFixed(6)}/${mRef.sharp.toFixed(6)} ` +
        `bright ${bright.toFixed(6)}/${mRef.bright.toFixed(6)} total ${total.toFixed(6)}/${ref.scores[pid][fiS].toFixed(6)}`);
    }
    perScore.set(fi, total);
  }
  jsScores.set(+pid, perScore);
}
check('scoring 全部指标与 Python 一致（容差 5e-6）', scoreFails === 0);

// ---- 2. tracking：关联结果与 Python 一致 ----
console.log('[2] tracking');
const rebuilt = ref.framesFaces.map((faces) => faces.map((fj) => {
  const f = rebuildFace(fj);
  f.personId = null;
  return f;
}));
associate(rebuilt);
let trackOk = true;
for (let fi = 0; fi < ref.framesFaces.length; fi++) {
  ref.framesFaces[fi].forEach((fj, k) => {
    if (rebuilt[fi][k].personId !== fj.personId) {
      trackOk = false;
      console.log(`  帧${fi} 第${k}脸 JS=${rebuilt[fi][k].personId} Python=${fj.personId}`);
    }
  });
}
check('tracking 关联与 Python 一致', trackOk);

// ---- 3. umeyama：相似变换与 Python LMEDS 一致（映射偏差 < 1px）----
console.log('[3] umeyama vs estimateAffinePartial2D(LMEDS)');
for (const sd of ref.swap_data) {
  if (sd.kind !== 'face_swap' || !sd.M) continue;
  const Mjs = estimateSimilarityTransform(sd.sp, sd.dp);
  check(`P${pidPlaceholder(sd)} M 存在`, !!Mjs);
  if (!Mjs) continue;
  // 映射偏差：把每个锚点分别用两个 M 映射，比较结果距离
  let maxErr = 0;
  for (const [sx, sy] of sd.sp) {
    const pj = [Mjs[0] * sx + Mjs[1] * sy + Mjs[2], Mjs[3] * sx + Mjs[4] * sy + Mjs[5]];
    const pp = [sd.M[0][0] * sx + sd.M[0][1] * sy + sd.M[0][2],
                sd.M[1][0] * sx + sd.M[1][1] * sy + sd.M[1][2]];
    maxErr = Math.max(maxErr, Math.hypot(pj[0] - pp[0], pj[1] - pp[1]));
  }
  check(`P${sd.pid} 映射最大偏差 ${maxErr.toFixed(4)}px`, maxErr < 1.0);
}
function pidPlaceholder(sd) { return sd.pid; }

// ---- 4. 决策流：基准帧 + rows 与 Python 一致 ----
console.log('[4] 决策流');
check(`基准帧 ${ref.base_idx}`,
  (() => {
    const nFrames = ref.names.length;
    const totals = new Array(nFrames).fill(0);
    for (let fi = 0; fi < nFrames; fi++)
      for (const per of jsScores.values()) totals[fi] += per.get(fi) || 0;
    let bi = 0;
    for (let fi = 1; fi < nFrames; fi++) if (totals[fi] > totals[bi]) bi = fi;
    return bi === ref.base_idx;
  })());
for (const r of ref.rows) {
  const perScore = jsScores.get(r.personId);
  const baseScore = perScore.get(ref.base_idx) || null;
  let bestFi = -1, bestV = -Infinity;
  for (const [fi, v] of perScore) if (v > bestV) { bestV = v; bestFi = fi; }
  if (baseScore === null) {
    check(`P${r.personId} swapped(missing) src=${r.srcFrame}`, bestFi === r.srcFrame && r.action === 'swapped(missing)');
    continue;
  }
  let maxEar = 0;
  const perRef = ref.raw[r.personId];
  for (const m of Object.values(perRef)) if (m.ear > maxEar) maxEar = m.ear;
  const earBase = perRef[ref.base_idx].ear;
  const eyesBase = eyesOpenScore(earBase, maxEar);
  const need = eyesBase < 0.72;
  const worth = bestV - baseScore > 0.05;
  const expected = (bestFi !== ref.base_idx && need && worth)
    ? 'swapped' : 'kept';
  const pyAction = r.action === 'swapped(missing)' ? 'swapped' : r.action;
  if (expected === 'swapped') {
    // swapped 行的 srcFrame 才记录 best 帧；并列帧排序对 1e-6 级浮点噪声敏感
    // （Python 自身也不稳定），两帧 Python 总分差 < 1e-4 视为等价。
    const tie = bestFi !== r.srcFrame
      && Math.abs(perScore.get(bestFi) - (ref.scores[r.personId]?.[r.srcFrame] ?? 0)) < 1e-4;
    check(`P${r.personId} swapped src=${r.srcFrame}（JS best=${bestFi}${tie ? '，并列帧' : ''}）`,
          pyAction === 'swapped' && (bestFi === r.srcFrame || tie));
  } else {
    // kept 行的 srcFrame 恒为 base_idx（Python 不记录 best 帧），只对齐动作
    check(`P${r.personId} kept`, pyAction === 'kept');
  }
}

console.log(`\n==== ${pass} 通过 / ${fail} 失败 ====`);
process.exit(fail ? 1 : 0);
