"""mediapipe FaceLandmarker 封装：一张图上检测所有人脸并提取 478 点关键点（含虹膜）。

478 点 = 468 基础网格 + 5 右虹膜(468-472) + 5 左虹膜(473-477)。
所有索引均指 mediapipe 标准人脸网格。
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# —— EAR（眼睛纵横比）六点：眼角-上下眼睑 ——
LEFT_EYE_EAR = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_EAR = (362, 385, 387, 263, 373, 380)

# —— 完整眼周轮廓（闭眼合成 / 遮挡区域估计用）——
LEFT_EYE_CONTOUR = (33, 246, 161, 160, 159, 158, 157, 173, 133,
                    155, 154, 153, 145, 144, 163, 7)
RIGHT_EYE_CONTOUR = (362, 398, 384, 385, 386, 387, 388, 466, 263,
                     249, 390, 373, 374, 380, 381)

IRIS_L = 473  # 左虹膜中心
IRIS_R = 468  # 右虹膜中心

# —— 对齐替换用的稳定锚点：外/内眼角、鼻尖、嘴角、下巴 ——
ALIGN_ANCHORS = (33, 263, 133, 362, 1, 61, 291, 152)

# —— 脸部椭圆（FACE_OVAL）关键点 ——
FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)

# 默认模型路径：相对本文件解析到项目根/models，不依赖运行目录（cwd）
DEFAULT_MODEL_PATH = (Path(__file__).resolve().parent.parent
                      / "models" / "face_landmarker.task")


@dataclass
class Face:
    """一帧中的一张人脸：像素坐标关键点 + 派生几何量。"""

    pts: np.ndarray              # (478, 2) float32，像素坐标
    frame_idx: int               # 所在帧序号
    person_id: int | None = None  # 跨帧关联后填入
    head_yaw: float | None = None   # 度，正对镜头时≈0（近似值）
    head_pitch: float | None = None

    @property
    def box(self) -> tuple[int, int, int, int]:
        """由关键点包围盒得到人脸框 (x, y, w, h)。"""
        x0, y0 = self.pts.min(axis=0)
        x1, y1 = self.pts.max(axis=0)
        return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))

    @property
    def center(self) -> np.ndarray:
        return self.pts.mean(axis=0)

    @property
    def face_width(self) -> float:
        x0, _, w, _ = self.box
        del x0
        return float(w)

    @property
    def ipd(self) -> float:
        """两外眼角距离，作为尺度归一化基准。"""
        return float(np.linalg.norm(self.pts[33] - self.pts[263]))


def _repack_stored(data: bytes) -> bytes:
    """用 Python zipfile 重打包为无压缩 zip。

    部分下载通道会在 zip 前混入杂字节，且个别 mediapipe 版本对原始
    zip 元数据兼容性差——统一重打包最稳。
    """
    with zipfile.ZipFile(io.BytesIO(data)) as zin:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zout:
            for item in zin.infolist():
                zout.writestr(item.filename, zin.read(item.filename))
        return buf.getvalue()


class Landmarker:
    """FaceLandmarker 的薄封装（IMAGE 模式，端侧推理）。"""

    def __init__(self, model_path: str | Path | None = None,
                 num_faces: int = 30):
        model_path = DEFAULT_MODEL_PATH if model_path is None else Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"找不到模型 {model_path}，请先下载 face_landmarker.task")
        data = model_path.read_bytes()
        i = data.find(b"PK\x03\x04")     # 裁掉 zip 前的杂字节（若有）
        if i > 0:
            data = data[i:]
        try:
            self._lm = self._create(data, num_faces)
        except RuntimeError:
            self._lm = self._create(_repack_stored(data), num_faces)

    @staticmethod
    def _create(model_bytes: bytes, num_faces: int):
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=num_faces,
            output_facial_transformation_matrixes=True,
        )
        return vision.FaceLandmarker.create_from_options(opts)

    def detect(self, bgr: np.ndarray) -> list[Face]:
        """输入 BGR 图，输出人脸列表（按 x 坐标排序，便于人读）。"""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(rgb))
        result = self._lm.detect(image)

        faces: list[Face] = []
        for i, lms in enumerate(result.face_landmarks):
            pts = np.empty((len(lms), 2), np.float32)
            pts[:, 0] = [p.x * w for p in lms]
            pts[:, 1] = [p.y * h for p in lms]
            face = Face(pts=pts, frame_idx=-1)

            # 头部姿态：变换矩阵旋转部分的第三列 = 人脸法向量（相机坐标系），
            # 正对镜头时法向≈光轴，yaw/pitch≈0（演示用近似，只取幅值也够用）
            if result.facial_transformation_matrixes:
                m = np.asarray(result.facial_transformation_matrixes[i])
                R = m[:3, :3]
                if abs(np.linalg.det(R)) > 1e-6:
                    n = R[:, 2] / (np.linalg.norm(R[:, 2]) + 1e-6)
                    face.head_yaw = float(np.degrees(np.arctan2(n[0], n[2])))
                    face.head_pitch = float(np.degrees(
                        np.arcsin(np.clip(n[1], -1.0, 1.0))))
            faces.append(face)

        faces.sort(key=lambda f: float(f.center[0]))
        return faces
