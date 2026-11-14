// 人脸替换合成（pipeline.py 的 _swap_face / _blend_person / _warp_to 移植）。
// 颜色流程与 Python 数值同轨：canvas RGBA → RGBA2BGR（等价 cv2.imread 的 BGR）
// → 全部按 Python 的 BGR 流程（BGR2LAB 等）→ 展示前 BGR2RGBA。
// 相似变换用手写 Umeyama（官方 opencv.js 无 estimateAffinePartial2D）。

import { estimateSimilarityTransform } from './umeyama.js';
import { ALIGN_ANCHORS } from './mp-landmarks.js';

// 颜色转换数字枚举（OpenCV imgproc 稳定值；@techstark 5.0 构建未导出 COLOR_* 常量）
const COLOR_RGBA2BGR = 3, COLOR_BGR2GRAY = 6, COLOR_BGR2LAB = 44, COLOR_LAB2BGR = 56;

export const MERGE_CONFIG = {
  earOpenRatio: 0.72,   // 基准帧中低于此睁眼分的人必须被替换
  minGain: 0.05,        // 替换带来的总分提升下限（最小替换原则）
  feather: 0.12,        // 融合羽化宽度（相对脸宽）
  colorMatch: true,     // 是否做 LAB 颜色迁移
};

export const FACE_GRAY_PAD = 1.0;      // _face_gray 的 pad
export const MAX_SIDE = 2048;          // 检测/评分用的长边上限

// 灰度人脸 ROI（_face_gray 移植）：box 外扩 10%，裁剪出 {data,w,h}
export function faceGray(gray, face) {
  const { x, y, w, h } = face.box;
  const p = FACE_GRAY_PAD;
  const x0 = Math.max(0, x - Math.trunc(w * p * 0.1));
  const y0 = Math.max(0, y - Math.trunc(h * p * 0.1));
  const x1 = Math.min(gray.w, x + w + Math.trunc(w * p * 0.1));
  const y1 = Math.min(gray.h, y + h + Math.trunc(h * p * 0.1));
  const rw = x1 - x0, rh = y1 - y0;
  const out = new Uint8ClampedArray(rw * rh);
  for (let yy = 0; yy < rh; yy++) {
    for (let xx = 0; xx < rw; xx++) out[yy * rw + xx] = gray.data[(y0 + yy) * gray.w + (x0 + xx)];
  }
  return { data: out, w: rw, h: rh };
}

// _warp_to：按 2×3 仿射矩阵 M 把 srcMat 整帧 warp 到 dst 尺寸（返回新 Mat）
export function warpTo(dstW, dstH, srcMat, M) {
  const mMat = new cv.Mat(2, 3, cv.CV_64F);
  mMat.data64F.set(M);
  const out = new cv.Mat();
  cv.warpAffine(srcMat, out, mMat, new cv.Size(dstW, dstH),
                cv.INTER_LINEAR, cv.BORDER_REPLICATE);
  mMat.delete();
  return out;
}

// _blend_person：把已对齐到 base 坐标系的 warped 中 dstFace 区域贴入
// （椭圆羽化 mask + LAB 颜色迁移 + alpha 融合）。返回新 Mat（base 不变）。
export function blendPerson(baseMat, warpedMat, dstFace, cfg = MERGE_CONFIG) {
  const W = baseMat.cols, H = baseMat.rows;
  const { x, y, w, h } = dstFace.box;
  const cx = x + w / 2, cy = y + h / 2;
  const ax = Math.trunc(w * 0.72), ay = Math.trunc(h * 0.92);  // 椭圆轴

  // 1) 椭圆羽化 mask（略放大，覆盖发际线和脸缘）
  const mask = new cv.Mat(H, W, cv.CV_32FC1, new cv.Scalar(0, 0, 0, 0));
  cv.ellipse(mask, new cv.Point(Math.trunc(cx), Math.trunc(cy)),
             new cv.Size(ax, ay), 0, 0, 360, new cv.Scalar(1.0), -1);
  const k = Math.max(3, Math.trunc(Math.min(ax, ay) * cfg.feather) | 1);
  const blurred = new cv.Mat();
  cv.GaussianBlur(mask, blurred, new cv.Size(0, 0), k / 2.5);
  mask.delete();
  const m = blurred.data32F;   // H*W float32

  // 2) LAB 颜色迁移：把换入区域统计量对齐到底片，消除色温漂移
  let warped = warpedMat;
  if (cfg.colorMatch) {
    let selCount = 0;
    for (let i = 0; i < W * H; i++) if (m[i] > 0.5) selCount++;
    if (selCount > 100) {
      const dLab = new cv.Mat(), sLab = new cv.Mat();
      cv.cvtColor(baseMat, dLab, COLOR_BGR2LAB);
      cv.cvtColor(warpedMat, sLab, COLOR_BGR2LAB);
      const dd = dLab.data, ss = sLab.data;
      for (let c = 0; c < 3; c++) {
        let s1 = 0, s2 = 0, m1 = 0;
        for (let i = 0; i < W * H; i++) {
          if (m[i] > 0.5) { const v = dd[i * 3 + c]; s1 += v; s2 += v * v; m1 += ss[i * 3 + c]; }
        }
        const dm = s1 / selCount;
        const dv = s2 / selCount - dm * dm;             // numpy .std() 总体标准差
        const ddStd = Math.sqrt(Math.max(0, dv));
        let s1b = 0, s2b = 0;
        for (let i = 0; i < W * H; i++) {
          if (m[i] > 0.5) { const v = ss[i * 3 + c]; s1b += v; s2b += v * v; }
        }
        const sm = s1b / selCount;
        const sv = s2b / selCount - sm * sm;
        const sdStd = Math.sqrt(Math.max(0, sv));
        let gain = sdStd > 1e-3 ? ddStd / sdStd : 1.0;
        gain = Math.min(1.4, Math.max(0.7, gain));
        for (let i = 0; i < W * H; i++) {
          if (m[i] > 0.5) {
            const v = (ss[i * 3 + c] - sm) * gain + dm;
            ss[i * 3 + c] = Math.min(255, Math.max(0, v));
          }
        }
      }
      const back = new cv.Mat();
      cv.cvtColor(sLab, back, COLOR_LAB2BGR);
      warped = back;
      dLab.delete(); sLab.delete();
    }
  }

  // 3) 羽化融合：base*(1-m) + warped*m，uint8 截断（对齐 numpy astype）
  const out = new cv.Mat(H, W, cv.CV_8UC3);
  for (let i = 0; i < W * H; i++) {
    const w0 = m[i], inv = 1 - w0;
    out.data[i * 3]     = Math.trunc(baseMat.data[i * 3]     * inv + warped.data[i * 3]     * w0);
    out.data[i * 3 + 1] = Math.trunc(baseMat.data[i * 3 + 1] * inv + warped.data[i * 3 + 1] * w0);
    out.data[i * 3 + 2] = Math.trunc(baseMat.data[i * 3 + 2] * inv + warped.data[i * 3 + 2] * w0);
  }
  if (warped !== warpedMat) warped.delete();
  blurred.delete();
  return out;
}

// _swap_face：把 srcMat 中 srcFace 的人贴回 baseMat 的 dstFace 位置
// （锚点相似变换 + 颜色迁移 + 羽化融合）。
export function swapFace(baseMat, dstFace, srcMat, srcFace, cfg = MERGE_CONFIG) {
  const sp = ALIGN_ANCHORS.map((i) => srcFace.pts[i]);
  const dp = ALIGN_ANCHORS.map((i) => dstFace.pts[i]);
  const M = estimateSimilarityTransform(sp, dp);
  if (!M) return baseMat;
  const warped = warpTo(baseMat.cols, baseMat.rows, srcMat, M);
  const out = blendPerson(baseMat, warped, dstFace, cfg);
  warped.delete();
  return out;
}

// BGR Mat → canvas（手写像素转换，不依赖 imshow/COLOR 枚举）
export function matToCanvas(bgrMat, canvasEl) {
  const w = bgrMat.cols, h = bgrMat.rows;
  canvasEl.width = w; canvasEl.height = h;
  const ctx = canvasEl.getContext('2d');
  const img = ctx.createImageData(w, h);
  const d = bgrMat.data, o = img.data;
  for (let i = 0; i < w * h; i++) {
    o[i * 4] = d[i * 3 + 2];      // R
    o[i * 4 + 1] = d[i * 3 + 1];  // G
    o[i * 4 + 2] = d[i * 3];      // B
    o[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
