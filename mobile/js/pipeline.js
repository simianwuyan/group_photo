// 合并管线（pipeline.py 的 merge_burst / _frame_shift 移植）。
// 参数逐字对齐 Python：MAX_SIDE=2048、耳阈 0.72、增益 0.05、缺人补回。

import { associate } from './tracking.js';
import { earValue, eyesOpenScore, smileScore, gazeScore,
         sharpnessScore, brightnessScore, totalScore } from './scoring.js';
import { estimateSimilarityTransform } from './umeyama.js';
import { ALIGN_ANCHORS, makeFace } from './mp-landmarks.js';
import { MERGE_CONFIG, MAX_SIDE, faceGray, warpTo, blendPerson, swapFace } from './compositing.js';

// ImageBitmap → { source: 缩放后的画布（供检测）, bgr: OpenCV BGR Mat }
// （等价 cv2.imread 的 BGR 布局）
function bitmapToPair(bitmap) {
  const scale = MAX_SIDE / Math.max(bitmap.width, bitmap.height);
  let w = bitmap.width, h = bitmap.height;
  const cnv = new OffscreenCanvas(w, h);
  let ctx = cnv.getContext('2d', { willReadFrequently: true });
  if (scale < 1) {
    w = Math.round(bitmap.width * scale); h = Math.round(bitmap.height * scale);
    cnv.width = w; cnv.height = h;
    ctx = cnv.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0, w, h);
  } else {
    ctx.drawImage(bitmap, 0, 0);
  }
  const rgba = cv.matFromImageData(ctx.getImageData(0, 0, w, h));
  const bgr = new cv.Mat();
  cv.cvtColor(rgba, bgr, cv.COLOR_RGBA2BGR);
  rgba.delete();
  return { source: cnv, bgr };
}

function faceFromMatGray(grayMat, face) {
  const data = new Uint8ClampedArray(grayMat.data);   // 拷贝 ROI 由 faceGray 完成
  return faceGray({ data, w: grayMat.cols, h: grayMat.rows }, face);
}

// _frame_shift：用两帧共同人脸的锚点估计整帧全局相似变换（src → 基准坐标系）
function frameShift(framesFaces, baseIdx, srcFi) {
  const baseByPid = new Map();
  for (const f of framesFaces[baseIdx]) {
    if (f.personId !== null && f.personId !== undefined) baseByPid.set(f.personId, f);
  }
  const sp = [], dp = [];
  for (const f of framesFaces[srcFi]) {
    const dst = baseByPid.get(f.personId);
    if (!dst) continue;
    for (const i of ALIGN_ANCHORS) { sp.push(f.pts[i]); dp.push(dst.pts[i]); }
  }
  if (sp.length === 0) return null;
  return estimateSimilarityTransform(sp, dp);
}

// 主入口：images = [{source: ImageBitmap/canvas, name}...]（按时间顺序）
// onProgress(stage, frac) 可选，用于 UI 进度。
export async function mergeBurst(images, landmarker, cfg = MERGE_CONFIG, onProgress = () => {}) {
  const names = images.map((im) => im.name);
  onProgress('load', 0);
  await tick();

  // —— 1. 载入（长边上限）——
  const pairs = images.map((im) => bitmapToPair(im.source));
  const framesBgr = pairs.map((p) => p.bgr);
  onProgress('load', 1);

  // —— 2. 检测 + 关联 ——
  const framesFaces = [];
  for (let i = 0; i < pairs.length; i++) {
    const { source } = pairs[i];
    const faces = landmarker.detectFaces(source, source.width, source.height, i);
    framesFaces.push(faces);
    onProgress('detect', (i + 1) / pairs.length);
    await tick();
  }
  associate(framesFaces);

  // —— 3. 逐人逐帧原始指标 ——
  const grayMats = framesBgr.map((f) => {
    const g = new cv.Mat();
    cv.cvtColor(f, g, cv.COLOR_BGR2GRAY);
    return g;
  });
  const raw = new Map();   // pid -> Map(fi -> metrics)
  for (let i = 0; i < framesFaces.length; i++) {
    for (const face of framesFaces[i]) {
      const pid = face.personId;
      const roi = faceFromMatGray(grayMats[i], face);
      const m = {
        ear: earValue(face),
        sharp: sharpnessScore(roi),
        bright: brightnessScore(roi),
        smile: smileScore(face),
        gaze: gazeScore(face),
      };
      if (!raw.has(pid)) raw.set(pid, new Map());
      raw.get(pid).set(i, m);
    }
    onProgress('score', (i + 1) / framesFaces.length);
    await tick();
  }

  // —— 4. 归一化睁眼分 + 总分 ——
  const scores = new Map();   // pid -> Map(fi -> total)
  for (const [pid, perFrame] of raw) {
    let maxEar = 0;
    for (const m of perFrame.values()) if (m.ear > maxEar) maxEar = m.ear;
    const sm = new Map();
    for (const [fi, m] of perFrame) {
      sm.set(fi, totalScore(eyesOpenScore(m.ear, maxEar), m.sharp, m.bright, m.smile, m.gaze));
    }
    scores.set(pid, sm);
  }
  const personIds = [...raw.keys()].sort((a, b) => a - b);
  if (personIds.length === 0) throw new Error('连拍中没有检测到人脸');

  // —— 5. 基准底片：在场人数加权的全局总分（缺人隐式扣分）——
  const nFrames = framesBgr.length;
  const frameTotal = new Array(nFrames).fill(0);
  for (let fi = 0; fi < nFrames; fi++) {
    for (const pid of personIds) frameTotal[fi] += scores.get(pid).get(fi) || 0;
  }
  let baseIdx = 0;
  for (let fi = 1; fi < nFrames; fi++) if (frameTotal[fi] > frameTotal[baseIdx]) baseIdx = fi;

  // —— 6. 逐人替换决策（最小替换原则）——
  const rows = [];          // PersonRow
  const swapJobs = [];      // [pid, srcFrame]
  for (const pid of personIds) {
    const perScores = scores.get(pid);
    const perRaw = raw.get(pid);
    const baseScore = perScores.has(baseIdx) ? perScores.get(baseIdx) : null;
    if (baseScore === null) {
      // 基准帧里没这个人 → 从其最佳帧整脸补回
      let bestFi = -1, bestV = -Infinity;
      for (const [fi, v] of perScores) if (v > bestV) { bestV = v; bestFi = fi; }
      swapJobs.push([pid, bestFi]);
      rows.push({ personId: pid, baseFrame: baseIdx, srcFrame: bestFi,
                  action: 'swapped(missing)', earBase: 0, earSrc: 0, scoreBase: 0, scoreSrc: 0 });
      continue;
    }
    let bestFi = -1, bestV = -Infinity;
    for (const [fi, v] of perScores) if (v > bestV) { bestV = v; bestFi = fi; }
    const earBase = perRaw.get(baseIdx).ear;
    let maxEar = 0;
    for (const m of perRaw.values()) if (m.ear > maxEar) maxEar = m.ear;
    const eyesBase = eyesOpenScore(earBase, maxEar);
    const need = eyesBase < cfg.earOpenRatio;                            // 状态不达标
    const worth = perScores.get(bestFi) - baseScore > cfg.minGain;       // 提升够大
    if (bestFi !== baseIdx && need && worth) {
      swapJobs.push([pid, bestFi]);
      rows.push({ personId: pid, baseFrame: baseIdx, srcFrame: bestFi, action: 'swapped',
                  earBase, earSrc: perRaw.get(bestFi).ear,
                  scoreBase: baseScore, scoreSrc: perScores.get(bestFi) });
    } else {
      rows.push({ personId: pid, baseFrame: baseIdx, srcFrame: baseIdx, action: 'kept',
                  earBase, earSrc: earBase, scoreBase: baseScore, scoreSrc: baseScore });
    }
  }

  // —— 7. 执行替换 ——
  onProgress('merge', 0);
  await tick();
  let base = framesBgr[baseIdx].clone();
  for (const [pid, srcFi] of swapJobs) {
    const dstFace = framesFaces[baseIdx].find((f) => f.personId === pid) || null;
    const srcFace = framesFaces[srcFi].find((f) => f.personId === pid);
    if (!dstFace) {
      // 基准帧缺人：用共同人脸估计整帧全局对齐，按变换后位置把该人贴回
      let M = frameShift(framesFaces, baseIdx, srcFi);
      if (!M) {
        M = [1, 0, 0, 0, 1, 0];   // 两帧无共同人脸：手持抖动很小，按原位置直接贴回
        console.warn(`[warn] P${pid} 基准帧与来源帧无共同人脸，按原位置直接贴回`);
      }
      const warped = warpTo(base.cols, base.rows, framesBgr[srcFi], M);
      // 锚点随全局变换定位：pts' = pts @ M[:, :2].T + M[:, 2]
      const pts = srcFace.pts.map(([px, py]) =>
        [M[0] * px + M[1] * py + M[2], M[3] * px + M[4] * py + M[5]]);
      const moved = makeFace(pts, srcFace.frameIdx);
      const blended = blendPerson(base, warped, moved, cfg);
      warped.delete(); base.delete(); base = blended;
      continue;
    }
    const out = swapFace(base, dstFace, framesBgr[srcFi], srcFace, cfg);
    base.delete(); base = out;
    onProgress('merge', 1);
    await tick();
  }

  // —— 8. 标注图 ——
  const annotated = base.clone();
  for (const r of rows) {
    let face = framesFaces[baseIdx].find((f) => f.personId === r.personId) || null;
    if (!face) face = framesFaces[r.srcFrame].find((f) => f.personId === r.personId);
    const { x, y, w, h } = face.box;
    const color = r.action === 'kept' ? [80, 190, 80] : [60, 140, 255];   // BGR：绿=保留 橙=替换
    cv.rectangle(annotated, new cv.Point(x, y), new cv.Point(x + w, y + h), color, 2);
    const label = `P${r.personId} ${r.action}`
      + (r.action.startsWith('swapped') ? ` <- ${names[r.srcFrame]}` : '');
    cv.putText(annotated, label, new cv.Point(x, Math.max(14, y - 6)),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA);
  }

  // 清理中间 Mat（保留 base/annotated 供展示）
  for (const g of grayMats) g.delete();
  for (const f of framesBgr) if (f !== base && f !== annotated) f.delete();

  onProgress('done', 1);
  return { merged: base, annotated, baseFrameIdx: baseIdx, rows,
           names, personIds, raw, scores };
}

const tick = () => new Promise((r) => setTimeout(r, 0));
