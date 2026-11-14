"""逐人逐帧质量评分：睁眼 / 清晰度 / 亮度 / 表情 / 视线。

原则：能用几何量（关键点）算的就不上模型——快、可解释、端侧零成本。
睁眼分数依赖"同一个人跨帧归一化"（pipeline 里做），这里提供原始量。
"""

from __future__ import annotations

import cv2
import numpy as np

from .landmarks import (Face, IRIS_L, IRIS_R, LEFT_EYE_EAR, RIGHT_EYE_EAR)

# EAR 的经验阈值：低于该值基本可判定闭眼（人脸尺度归一后与脸大小无关）
EAR_BLINK_ABS = 0.14
# 相对阈值：ear / 该人全场最大 EAR 低于此值判为闭眼（消除个体眼型差异）
EAR_RATIO_OPEN = 0.72


def ear_value(face: Face) -> float:
    """眼睛纵横比：双眼均值，越小越闭。"""

    def one_eye(idx6: tuple[int, ...]) -> float:
        p = face.pts[list(idx6)]
        v1 = np.linalg.norm(p[1] - p[5])
        v2 = np.linalg.norm(p[2] - p[4])
        h = np.linalg.norm(p[0] - p[3])
        return float((v1 + v2) / (2 * h + 1e-6))

    return (one_eye(LEFT_EYE_EAR) + one_eye(RIGHT_EYE_EAR)) / 2


def eyes_open_score(ear: float, person_max_ear: float) -> float:
    """睁眼分数 ∈ [0,1]：绝对阈值 + 相对阈值双保险。"""
    denom = max(person_max_ear, EAR_BLINK_ABS)
    ratio = ear / denom
    # ratio 在 [EAR_RATIO_OPEN, 1+] 映射到 [0.5, 1]，低于则快速衰减
    if ratio >= EAR_RATIO_OPEN:
        return float(np.clip(0.5 + 0.5 * (ratio - EAR_RATIO_OPEN)
                             / (1 - EAR_RATIO_OPEN + 1e-6), 0.5, 1.0))
    return float(np.clip(ratio / (2 * EAR_RATIO_OPEN), 0.0, 0.5))


def smile_score(face: Face) -> float:
    """微笑分数：嘴角间距相对瞳距 + 嘴角相对唇中心上提量。"""
    ipd = face.ipd
    mouth_w = np.linalg.norm(face.pts[61] - face.pts[291]) / ipd
    corner_y = (face.pts[61][1] + face.pts[291][1]) / 2
    lip_y = (face.pts[13][1] + face.pts[14][1]) / 2
    lift = (lip_y - corner_y) / ipd  # 嘴角高于唇中心为正
    # 经验线性组合后压到 [0,1]（自然表情≈0.5，笑≈0.8+）
    s = 0.5 * np.clip((mouth_w - 0.28) / 0.25, 0, 1) \
        + 0.5 * np.clip((lift + 0.02) / 0.12, 0, 1)
    return float(np.clip(s, 0, 1))


def gaze_score(face: Face) -> float:
    """视线/朝向分数：虹膜居中程度 + 头部正对程度。"""
    def iris_centering(iris: int, outer: int, inner: int) -> float:
        eye_mid = (face.pts[outer] + face.pts[inner]) / 2
        eye_w = np.linalg.norm(face.pts[outer] - face.pts[inner]) + 1e-6
        off = np.linalg.norm(face.pts[iris] - eye_mid) / eye_w
        return float(np.clip(1.0 - off * 1.6, 0, 1))

    c = (iris_centering(IRIS_L, 33, 133) + iris_centering(IRIS_R, 362, 263)) / 2
    if face.head_yaw is not None:
        # 偏航/俯仰每偏 15° 扣一档
        ang = np.hypot(face.head_yaw, face.head_pitch)
        head = float(np.exp(-(ang / 18.0) ** 2))
        return 0.6 * c + 0.4 * head
    return c


def sharpness_score(gray_face: np.ndarray) -> float:
    """清晰度：人脸区域 Laplacian 方差，映射到 [0,1]。"""
    v = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    return float(np.clip(v / 250.0, 0, 1))


def brightness_score(gray_face: np.ndarray) -> float:
    """亮度：均值过暗/过曝都扣分，中间调得 1。"""
    m = float(gray_face.mean()) / 255.0
    return float(np.exp(-((m - 0.55) / 0.28) ** 2))


def total_score(eyes: float, sharp: float, bright: float,
                smile: float, gaze: float,
                w=(0.45, 0.20, 0.10, 0.15, 0.10)) -> float:
    """加权总分。睁眼权重最高——合影废片第一杀手。"""
    return float(np.dot([eyes, sharp, bright, smile, gaze], w))
