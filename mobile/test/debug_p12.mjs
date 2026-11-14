// 调试：打印 P1/P2 逐帧 JS total vs Python total（并列排序问题定位）
import { readFileSync } from 'node:fs';
import { earValue, eyesOpenScore, smileScore, gazeScore,
         sharpnessScore, brightnessScore, totalScore } from '../js/scoring.js';

const ref = JSON.parse(readFileSync(new URL(process.argv[2] || './ref.json', import.meta.url), 'utf-8'));

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
  return { pts, headYaw: fj.headYaw, headPitch: fj.headPitch,
    box: { x: Math.trunc(x0), y: Math.trunc(y0), w: Math.trunc(x1 - x0), h: Math.trunc(y1 - y0) },
    ipd: Math.hypot(pts[33][0] - pts[263][0], pts[33][1] - pts[263][1]) };
}

for (const pid of ['1', '2']) {
  const perRef = ref.raw[pid];
  if (!perRef) continue;
  let maxEar = 0;
  for (const m of Object.values(perRef)) if (m.ear > maxEar) maxEar = m.ear;
  console.log(`P${pid} maxEar=${maxEar.toFixed(4)}`);
  for (const [fiS, mRef] of Object.entries(perRef)) {
    const face = rebuildFace(ref.framesFaces[+fiS].find((f) => f.personId === +pid));
    const ear = earValue(face);
    const eyes = eyesOpenScore(ear, maxEar);
    const sharp = sharpnessScore(ref.rois[`${pid}_${fiS}`]);
    const bright = brightnessScore(ref.rois[`${pid}_${fiS}`]);
    const smile = smileScore(face);
    const gaze = gazeScore(face);
    const total = totalScore(eyes, sharp, bright, smile, gaze);
    const py = ref.scores[pid][fiS];
    const d = (total - py).toExponential(2);
    const star = Math.abs(total - py) > 1e-4 ? ' <<< ' : '';
    console.log(`  f${fiS} js=${total.toFixed(6)} py=${py.toFixed(6)} d=${d} eyes=${eyes.toFixed(4)} sharp=${sharp.toFixed(4)} bright=${bright.toFixed(4)} smile=${smile.toFixed(4)} gaze=${gaze.toFixed(4)}${star}`);
  }
}
