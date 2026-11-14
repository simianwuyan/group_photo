// 跨帧身份关联（tracking.py 逐字移植）。
// 连拍间隔短（<0.5s）、人几乎不动，贪心 IoU 匹配已足够稳；
// 对漏检/新出现的脸做距离兜底。不上重模型（embedding），保持零成本。

export const IOU_MATCH = 0.25;     // IoU 低于此值不认为同一张脸
export const DIST_FALLBACK = 1.2;  // 中心距离 < min(脸宽)*此值 时兜底匹配

function iou(a, b) {
  const x0 = Math.max(a.x, b.x), y0 = Math.max(a.y, b.y);
  const x1 = Math.min(a.x + a.w, b.x + b.w), y1 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0.0;
}

// 就地写入 personId，帧必须按时间顺序排列。
// 排序细节对齐 Python：pairs 元组 (iou, pid, fi) reverse=True → 全降序。
export function associate(framesFaces) {
  let nextId = 0;
  const tracks = new Map();  // pid -> 上一帧的脸（Map 保持插入顺序）

  for (const faces of framesFaces) {
    let unmatched = faces.map((_, i) => i);

    // 1) 贪心 IoU：所有 (track, face) 对里挑 IoU 最大的先匹配
    const pairs = [];
    for (const [pid, last] of tracks) {
      for (const fi of unmatched) {
        const v = iou(last.box, faces[fi].box);
        if (v >= IOU_MATCH) pairs.push([v, pid, fi]);
      }
    }
    pairs.sort((p, q) => (q[0] - p[0]) || (q[1] - p[1]) || (q[2] - p[2]));
    const used = new Set();
    for (const [, pid, fi] of pairs) {
      if (unmatched.includes(fi) && !used.has(pid)) {
        faces[fi].personId = pid;
        unmatched = unmatched.filter((x) => x !== fi);
        used.add(pid);
      }
    }

    // 2) 距离兜底：相邻帧小位移漏检边界的情况
    for (const fi of [...unmatched]) {
      let bestPid = null, bestD = DIST_FALLBACK;
      for (const [pid, last] of tracks) {
        if (used.has(pid)) continue;
        const d = Math.hypot(faces[fi].center[0] - last.center[0],
                             faces[fi].center[1] - last.center[1])
                / Math.max(last.faceWidth, 1);
        if (d < bestD) { bestD = d; bestPid = pid; }
      }
      if (bestPid !== null) {
        faces[fi].personId = bestPid;
        unmatched = unmatched.filter((x) => x !== fi);
        used.add(bestPid);
      }
    }

    // 3) 剩下的 = 新出现的人
    for (const fi of unmatched) {
      faces[fi].personId = nextId;
      nextId++;
    }

    // 更新轨迹（保留最后观测）
    for (const f of faces) tracks.set(f.personId, f);
  }
  return framesFaces;
}
