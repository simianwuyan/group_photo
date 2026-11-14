"""合成连拍测试数据：从一张真实合影生成带"闭眼 + 手持抖动"的连拍序列。

用途：在没有真机连拍样本时验证管线，且自带 ground truth 便于自动验证。
流程：每帧 = 原图 -> 随机若干人闭眼(皮肤色填充眼睑) -> 全局仿射抖动 -> 亮度微扰。
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np

from .landmarks import (Face, Landmarker, LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR)
from .pipeline import imread_exif


def close_eyes(canvas: np.ndarray, face: Face) -> None:
    """在 canvas 上把一张脸的眼睛'画闭'：皮肤色填充眼裂 + 上眼睑折痕。"""
    for contour in (LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR):
        pts = face.pts[list(contour)].astype(np.int32)
        hull = cv2.convexHull(pts)
        x, y, w, h = cv2.boundingRect(pts)

        # 从眼下方脸颊采样皮肤色（中位数抗噪）
        sy = min(canvas.shape[0] - 1, y + h + int(0.9 * h) + 3)
        patch = canvas[max(0, sy - 2):sy + 4, max(0, x):x + max(w, 4)]
        if patch.size == 0:
            continue
        skin = np.median(patch.reshape(-1, 3), axis=0)

        # 填充眼裂
        layer = canvas.copy()
        cv2.fillConvexPoly(layer, hull, tuple(int(c) for c in skin))
        # 只在眼周小范围内混合，避免影响其他区域
        roi = np.zeros(canvas.shape[:2], np.uint8)
        cv2.fillConvexPoly(roi, hull, 255)
        roi = cv2.dilate(roi, np.ones((5, 5), np.uint8))
        roi = cv2.GaussianBlur(roi, (0, 0), 2.5)
        m = roi.astype(float)[..., None] / 255.0
        canvas[:] = (canvas * (1 - m) + layer * m).astype(np.uint8)

        # 上眼睑折痕（比皮肤略暗的弧线）
        upper = pts[[0, 2, 4, 6, 8, 10, 12, 14]] if len(contour) >= 15 else pts
        crease = (upper - [0, int(0.25 * h)]).astype(np.int32)
        dark = tuple(int(c * 0.82) for c in skin)
        cv2.polylines(canvas, [crease], False, dark, 1, cv2.LINE_AA)


def make_burst(base_path: str | Path, out_dir: str | Path,
               n_frames: int = 12, blink_prob: float = 0.25,
               seed: int = 42) -> dict:
    """生成连拍序列 + ground truth，返回 GT 字典（同时存 out_dir/gt.json）。"""
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = imread_exif(base_path)
    assert img is not None, f"读图失败: {base_path}"
    lm = Landmarker()
    faces = lm.detect(img)

    # 每人随机指定闭眼帧集合（保证每人至少有一帧睁眼，否则没法救）
    gt: dict[str, dict[str, list[int]]] = {"persons": {}, "blink_frames": {}}
    for i, f in enumerate(faces):
        f.person_id = i
        n_blink = max(1, int(round(n_frames * blink_prob * rng.random() * 1.5))) \
            if rng.random() < 0.8 else 0  # 约 20% 的人全程不眨眼
        blink_frames = sorted(rng.sample(range(n_frames), min(n_blink, n_frames - 2))
                               ) if n_blink > 0 else []
        gt["persons"][str(i)] = [float(v) for v in f.box]
        gt["blink_frames"][str(i)] = blink_frames

    h, w = img.shape[:2]
    for k in range(n_frames):
        canvas = img.copy()
        # 先闭眼（原图坐标系），再统一做全局抖动
        for pid, bfs in gt["blink_frames"].items():
            if k in bfs:
                close_eyes(canvas, faces[int(pid)])

        # 手持抖动：小平移 + 微旋转 + 微缩放
        ang = rng.uniform(-0.4, 0.4)
        s = 1 + rng.uniform(-0.002, 0.002)
        tx, ty = rng.uniform(-6, 6), rng.uniform(-4, 4)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, s)
        M[:, 2] += (tx, ty)
        canvas = cv2.warpAffine(canvas, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        canvas = np.clip(canvas.astype(np.int16) + rng.randint(-6, 6),
                         0, 255).astype(np.uint8)  # 亮度微扰
        cv2.imwrite(str(out_dir / f"frame_{k:03d}.jpg"), canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

    (out_dir / "gt.json").write_text(json.dumps(gt, indent=2), "utf-8")
    return gt
