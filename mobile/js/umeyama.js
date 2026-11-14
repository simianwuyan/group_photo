// 2D 相似变换最小二乘（dst = s·R·src + t，复数域闭式解，替代被官方 opencv.js
// 裁剪的 estimateAffinePartial2D）。8 锚点为同一张脸的关键点，质量均匀；
// 附一次"离群重拟合"（残差 > 3×中位绝对残差的点剔除后重算），近似 LMEDS 行为。
// 返回 2×3 行主序矩阵 M = [a, -b, tx; b, a, ty]（复数 w = a + bi）。

function fitOnce(src, dst) {
  const n = src.length;
  let muSx = 0, muSy = 0, muDx = 0, muDy = 0;
  for (let i = 0; i < n; i++) {
    muSx += src[i][0]; muSy += src[i][1];
    muDx += dst[i][0]; muDy += dst[i][1];
  }
  muSx /= n; muSy /= n; muDx /= n; muDy /= n;

  // w = Σ conj(srcC)·dstC / Σ|srcC|²（复数最小二乘）
  let wre = 0, wim = 0, ss = 0;
  for (let i = 0; i < n; i++) {
    const sx = src[i][0] - muSx, sy = src[i][1] - muSy;
    const dx = dst[i][0] - muDx, dy = dst[i][1] - muDy;
    wre += sx * dx + sy * dy;
    wim += sx * dy - sy * dx;
    ss += sx * sx + sy * sy;
  }
  if (ss < 1e-9) return null;
  wre /= ss; wim /= ss;

  const tx = muDx - (wre * muSx - wim * muSy);
  const ty = muDy - (wim * muSx + wre * muSy);
  return { m: [wre, -wim, tx, wim, wre, ty], wre, wim, tx, ty };
}

const applyM = (m, p) => [m[0] * p[0] + m[1] * p[1] + m[2],
                          m[3] * p[0] + m[4] * p[1] + m[5]];

// 返回 2×3 矩阵数组；点数不足或退化返回 null。
export function estimateSimilarityTransform(src, dst) {
  if (!src || !dst || src.length !== dst.length || src.length < 2) return null;

  let fit = fitOnce(src, dst);
  if (!fit) return null;

  // 一次离群重拟合：残差中位绝对值的 3 倍视为离群（至少保留 4 点）
  if (src.length >= 6) {
    const res = src.map((_, i) => {
      const q = applyM(fit.m, src[i]);
      return Math.hypot(q[0] - dst[i][0], q[1] - dst[i][1]);
    });
    const sorted = [...res].sort((a, b) => a - b);
    const mid = sorted[Math.floor(sorted.length / 2)] || 0;
    const mad = mid * 3 + 1e-9;
    const keepS = [], keepD = [];
    for (let i = 0; i < src.length; i++) {
      if (res[i] <= mad || keepS.length < Math.max(4, src.length - 2)) {
        keepS.push(src[i]); keepD.push(dst[i]);
      }
    }
    if (keepS.length >= 2 && keepS.length < src.length) {
      const refit = fitOnce(keepS, keepD);
      if (refit) fit = refit;
    }
  }
  return fit.m;
}
