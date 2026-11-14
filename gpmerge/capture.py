"""本机摄像头连拍采集：抓 n_frames 帧用于真实连拍合并（路演演示 / 采数据两用）。

真实连拍与合成数据的差异点（本模块只负责采集，处理仍走 gpmerge.pipeline）：
- 帧间有真实手持抖动与人物动作，跨度约 1.6 秒，足够跨过多次眨眼（100-400ms）；
- 预热丢弃前若干帧，等待自动曝光/白平衡收敛，避免首帧欠曝干扰亮度评分。
"""

from __future__ import annotations

import time

import cv2
import numpy as np


def capture_burst(camera_index: int = 0,
                  n_frames: int = 12,
                  interval_ms: int = 130,
                  warmup_ms: int = 600,
                  width: int = 1280,
                  height: int = 720) -> list[np.ndarray]:
    """打开摄像头，预热后按时间间隔抓 n_frames 帧 BGR 图。

    Windows 下优先 DirectShow 后端（启动快），失败回退默认后端。
    打不开摄像头或读帧中断均抛 RuntimeError。
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)   # 回退默认后端（MSMF 等）
    if not cap.isOpened():
        raise RuntimeError(f"打不开摄像头 {camera_index}（请检查编号与占用情况）")
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # 预热：丢帧等自动曝光/白平衡收敛
        warmup_end = time.time() + warmup_ms / 1000
        while time.time() < warmup_end:
            cap.read()

        frames: list[np.ndarray] = []
        next_t = 0.0
        while len(frames) < n_frames:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("摄像头读帧失败（连接中断或被其他程序占用）")
            now = time.time()
            if now >= next_t:
                frames.append(frame.copy())    # copy：避免底层 buffer 被下一帧覆盖
                next_t = now + interval_ms / 1000
        return frames
    finally:
        cap.release()
