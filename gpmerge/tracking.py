"""跨帧身份关联：把每一帧的人脸串成"同一个人"的轨迹。

连拍间隔短（<0.5s）、人几乎不动，贪心 IoU 匹配已足够稳；
对漏检/新出现的脸做距离兜底。不上重模型（embedding），保持零成本。
"""

from __future__ import annotations

import numpy as np

from .landmarks import Face

IOU_MATCH = 0.25          # IoU 低于此值不认为同一张脸
DIST_FALLBACK = 1.2       # 中心距离 < min(脸宽)*此值 时兜底匹配


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def associate(frames_faces: list[list[Face]]) -> list[list[Face]]:
    """就地写入 person_id，返回同样结构。帧必须按时间顺序排列。"""
    next_id = 0
    tracks: dict[int, Face] = {}  # pid -> 上一帧的脸（用于预测本帧位置）

    for faces in frames_faces:
        unmatched = list(range(len(faces)))
        # 1) 贪心 IoU：所有 (track, face) 对里挑 IoU 最大的先匹配
        pairs = []
        for pid, last in tracks.items():
            for fi in unmatched:
                iou = _iou(last.box, faces[fi].box)
                if iou >= IOU_MATCH:
                    pairs.append((iou, pid, fi))
        pairs.sort(reverse=True)
        used_pids: set[int] = set()
        for iou, pid, fi in pairs:
            if fi in unmatched and pid not in used_pids:
                faces[fi].person_id = pid
                unmatched.remove(fi)
                used_pids.add(pid)

        # 2) 距离兜底：相邻帧小位移漏检边界的情况
        for fi in list(unmatched):
            best_pid, best_d = None, DIST_FALLBACK
            for pid, last in tracks.items():
                if pid in used_pids:
                    continue
                d = (np.linalg.norm(faces[fi].center - last.center)
                     / max(last.face_width, 1))
                if d < best_d:
                    best_d, best_pid = d, pid
            if best_pid is not None:
                faces[fi].person_id = best_pid
                unmatched.remove(fi)
                used_pids.add(best_pid)

        # 3) 剩下的 = 新出现的人
        for fi in unmatched:
            faces[fi].person_id = next_id
            next_id += 1

        # 更新轨迹（保留最后观测）
        for f in faces:
            tracks[f.person_id] = f

    return frames_faces
