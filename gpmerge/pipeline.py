"""合并管线：连拍帧 -> 每人最佳状态的一张成片。

策略（对应产品设计三原则）：
1. 基准底片：全局总分最高的帧做底，背景/姿态 90% 一致；
2. 最小替换：只替换基准帧里"状态不达标"的人，替换越少痕迹越小；
3. 局部对齐：人脸锚点相似变换 + 椭圆羽化 + LAB 颜色迁移 + 拉普拉斯金字塔思想
   的羽化融合，不做全身重对齐（不可解问题）。
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import scoring
from .landmarks import ALIGN_ANCHORS, Face, Landmarker
from .tracking import associate

MAX_SIDE = 2048  # 检测/评分用的长边上限（成片也按此尺寸输出）


def imread_exif(p: str | Path) -> np.ndarray | None:
    """读图并确保按 EXIF 方向转正（手机连拍/竖拍必带方向标记）。

    注意：新版 OpenCV 的 imread/imdecode 默认已应用 EXIF 方向，旧版则不会；
    这里显式用 PIL 判定并转正（orientation != 1 时走 PIL 结果），任何版本下
    行为一致、不会双重旋转。无 EXIF 走 cv2 快路径，PIL 异常时兜底 cv2 解码。
    """
    data = Path(p).read_bytes()
    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(data)) as im:
            orientation = im.getexif().get(274, 1)   # 274 = Orientation
            if orientation != 1:
                fixed = ImageOps.exif_transpose(im).convert("RGB")
                return cv2.cvtColor(np.asarray(fixed), cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


@dataclass
class MergeConfig:
    ear_open_ratio: float = 0.72   # 基准帧中低于此睁眼分的人必须被替换
    min_gain: float = 0.05         # 替换带来的总分提升下限（最小替换原则）
    feather: float = 0.12          # 融合羽化宽度（相对脸宽）
    color_match: bool = True       # 是否做 LAB 颜色迁移


@dataclass
class PersonRow:
    person_id: int
    base_frame: int
    src_frame: int
    action: str                    # kept / swapped
    ear_base: float = 0.0
    ear_src: float = 0.0
    score_base: float = 0.0
    score_src: float = 0.0


@dataclass
class MergeResult:
    merged: np.ndarray
    annotated: np.ndarray          # 合并结果 + 框/标注
    base_frame_idx: int
    rows: list[PersonRow] = field(default_factory=list)


def _load_frames(paths: list[Path]) -> list[np.ndarray]:
    frames = []
    for p in paths:
        img = imread_exif(p)
        assert img is not None, f"读图失败: {p}"
        h, w = img.shape[:2]
        scale = MAX_SIDE / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        frames.append(img)
    return frames


def _face_gray(gray: np.ndarray, face: Face, pad: float = 1.0) -> np.ndarray:
    x, y, w, h = face.box
    x0, y0 = max(0, x - int(w * pad * 0.1)), max(0, y - int(h * pad * 0.1))
    x1 = min(gray.shape[1], x + w + int(w * pad * 0.1))
    y1 = min(gray.shape[0], y + h + int(h * pad * 0.1))
    return gray[y0:y1, x0:x1]


def _warp_to(dst_size: tuple[int, int], src: np.ndarray,
             M: np.ndarray) -> np.ndarray:
    """按仿射矩阵 M 把 src 整帧 warp 到 dst 尺寸坐标系。"""
    return cv2.warpAffine(src, M, dst_size, flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _blend_person(base: np.ndarray, warped: np.ndarray, dst_face: Face,
                  cfg: MergeConfig) -> np.ndarray:
    """把已对齐到 base 坐标系的 warped 中 dst_face 区域贴入（羽化 + 颜色迁移）。"""
    # 1) 人脸椭圆羽化 mask（略放大，覆盖发际线和脸缘）
    x, y, w, h = dst_face.box
    cx, cy = x + w / 2, y + h / 2
    axes = (int(w * 0.72), int(h * 0.92))
    mask = np.zeros(base.shape[:2], np.float32)
    cv2.ellipse(mask, (int(cx), int(cy)), axes, 0, 0, 360, 1.0, -1)
    k = max(3, int(min(axes) * cfg.feather) | 1)
    mask = cv2.GaussianBlur(mask, (0, 0), k / 2.5)

    # 2) LAB 颜色迁移：把换入区域统计量对齐到底片，消除色温漂移
    if cfg.color_match:
        sel = mask > 0.5
        if sel.sum() > 100:
            d_lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB).astype(np.float32)
            s_lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
            for c in range(3):
                dm, ds = d_lab[..., c][sel].mean(), s_lab[..., c][sel].mean()
                dd, sd = d_lab[..., c][sel].std(), s_lab[..., c][sel].std()
                gain = dd / sd if sd > 1e-3 else 1.0
                gain = float(np.clip(gain, 0.7, 1.4))
                s_lab[..., c] = (s_lab[..., c] - ds) * gain + dm
            warped = np.clip(s_lab, 0, 255).astype(np.uint8)
            warped = cv2.cvtColor(warped, cv2.COLOR_LAB2BGR)

    # 3) 羽化融合
    m = mask[..., None]
    return (base * (1 - m) + warped * m).astype(np.uint8)


def _swap_face(dst: np.ndarray, dst_face: Face,
               src: np.ndarray, src_face: Face,
               cfg: MergeConfig) -> np.ndarray:
    """把 src 中的这个人贴回 dst 的对应位置（对齐 + 颜色迁移 + 羽化融合）。"""
    # 锚点相似变换：src 人脸 -> dst 人脸坐标系
    sp = src_face.pts[list(ALIGN_ANCHORS)].astype(np.float32)
    dp = dst_face.pts[list(ALIGN_ANCHORS)].astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(sp, dp, method=cv2.LMEDS)
    if M is None:
        return dst
    warped = _warp_to((dst.shape[1], dst.shape[0]), src, M)
    return _blend_person(dst, warped, dst_face, cfg)


def _frame_shift(frames_faces: list[list[Face]], base_idx: int,
                 src_fi: int) -> np.ndarray | None:
    """用两帧共同人脸的锚点估计整帧全局相似变换（src -> 基准坐标系）。

    缺人补回时基准帧里没有这个人的"目标位置"，只能靠两帧共同人物
    把整帧画面先对齐；无共同人脸时返回 None（调用方按原位置兜底）。
    """
    base_by_pid = {f.person_id: f for f in frames_faces[base_idx]
                   if f.person_id is not None}
    sp, dp = [], []
    for f in frames_faces[src_fi]:
        dst = base_by_pid.get(f.person_id)
        if dst is None:
            continue
        sp.extend(f.pts[list(ALIGN_ANCHORS)].astype(np.float32))
        dp.extend(dst.pts[list(ALIGN_ANCHORS)].astype(np.float32))
    if not sp:
        return None
    M, _ = cv2.estimateAffinePartial2D(np.asarray(sp), np.asarray(dp),
                                       method=cv2.LMEDS)
    return M


def merge_burst(frame_paths: list[str | Path],
                model_path: str | Path | None = None,
                cfg: MergeConfig | None = None,
                landmarker: Landmarker | None = None) -> MergeResult:
    cfg = cfg or MergeConfig()
    paths = sorted(Path(p) for p in frame_paths)
    frames = _load_frames(paths)
    lm = landmarker or Landmarker(model_path)

    # —— 1. 检测 + 关联 ——
    frames_faces = [lm.detect(f) for f in frames]
    for i, ff in enumerate(frames_faces):
        for face in ff:
            face.frame_idx = i
    associate(frames_faces)

    # —— 2. 逐人逐帧原始指标 ——
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    raw: dict[int, dict[int, dict]] = {}   # pid -> {frame: metrics}
    for i, ff in enumerate(frames_faces):
        for face in ff:
            pid = face.person_id
            g = _face_gray(grays[i], face)
            raw.setdefault(pid, {})[i] = {
                "ear": scoring.ear_value(face),
                "sharp": scoring.sharpness_score(g),
                "bright": scoring.brightness_score(g),
                "smile": scoring.smile_score(face),
                "gaze": scoring.gaze_score(face),
            }

    # —— 3. 归一化睁眼分 + 总分 ——
    scores: dict[int, dict[int, float]] = {}
    for pid, per_frame in raw.items():
        max_ear = max(m["ear"] for m in per_frame.values())
        scores[pid] = {
            fi: scoring.total_score(
                eyes=scoring.eyes_open_score(m["ear"], max_ear),
                sharp=m["sharp"], bright=m["bright"],
                smile=m["smile"], gaze=m["gaze"])
            for fi, m in per_frame.items()
        }
    person_ids = sorted(raw)
    if not person_ids:
        raise RuntimeError("连拍中没有检测到人脸")

    # —— 4. 基准底片：在场人数加权的全局总分（缺人隐式扣分）——
    n_frames = len(frames)
    frame_total = np.zeros(n_frames)
    for fi in range(n_frames):
        frame_total[fi] = sum(scores[pid].get(fi, 0.0) for pid in person_ids)
    base_idx = int(np.argmax(frame_total))
    base = frames[base_idx].copy()

    # —— 5. 逐人替换决策（最小替换原则）——
    rows: list[PersonRow] = []
    swap_jobs: list[tuple[int, int]] = []  # (pid, src_frame)
    for pid in person_ids:
        base_score = scores[pid].get(base_idx)
        if base_score is None:            # 基准帧里没这个人 → 从其最佳帧整脸补回
            best_fi = max(scores[pid], key=scores[pid].get)
            swap_jobs.append((pid, best_fi))
            rows.append(PersonRow(pid, base_idx, best_fi, "swapped(missing)"))
            continue
        best_fi = max(scores[pid], key=scores[pid].get)
        base_face = next(f for f in frames_faces[base_idx] if f.person_id == pid)
        ear_base = raw[pid][base_idx]["ear"]
        max_ear = max(m["ear"] for m in raw[pid].values())
        eyes_base = scoring.eyes_open_score(ear_base, max_ear)
        need = eyes_base < cfg.ear_open_ratio          # 状态不达标
        worth = scores[pid][best_fi] - base_score > cfg.min_gain
        if best_fi != base_idx and need and worth:
            swap_jobs.append((pid, best_fi))
            rows.append(PersonRow(pid, base_idx, best_fi, "swapped",
                                  ear_base=ear_base,
                                  ear_src=raw[pid][best_fi]["ear"],
                                  score_base=base_score,
                                  score_src=scores[pid][best_fi]))
        else:
            rows.append(PersonRow(pid, base_idx, base_idx, "kept",
                                  ear_base=ear_base, ear_src=ear_base,
                                  score_base=base_score, score_src=base_score))

    # —— 6. 执行替换 ——
    for pid, src_fi in swap_jobs:
        dst_face = next((f for f in frames_faces[base_idx]
                         if f.person_id == pid), None)
        src_face = next(f for f in frames_faces[src_fi] if f.person_id == pid)
        if dst_face is None:
            # 基准帧缺人：用共同人脸估计整帧全局对齐，按变换后位置把该人贴回
            M = _frame_shift(frames_faces, base_idx, src_fi)
            if M is None:
                # 两帧无共同人脸（极罕见）：手持抖动很小，按原位置直接贴回
                M = np.eye(2, 3)
                print(f"[warn] P{pid} 基准帧与来源帧无共同人脸，按原位置直接贴回")
            warped = _warp_to((base.shape[1], base.shape[0]), frames[src_fi], M)
            pts = src_face.pts @ M[:, :2].T + M[:, 2]
            moved = Face(pts=pts.astype(np.float32), frame_idx=src_face.frame_idx)
            base = _blend_person(base, warped, moved, cfg)
            continue
        base = _swap_face(base, dst_face, frames[src_fi], src_face, cfg)

    # —— 7. 标注图 + 报告 ——
    annotated = base.copy()
    names = [Path(p).name for p in paths]
    for r in rows:
        face = next((f for f in frames_faces[base_idx] if f.person_id == r.person_id), None)
        if face is None:
            face = next(f for f in frames_faces[r.src_frame]
                        if f.person_id == r.person_id)
        x, y, w, h = face.box
        color = (80, 190, 80) if r.action == "kept" else (60, 140, 255)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        label = f"P{r.person_id} {r.action}" + (
            f" <- {names[r.src_frame]}" if r.action.startswith("swapped") else "")
        cv2.putText(annotated, label, (x, max(14, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return MergeResult(merged=base, annotated=annotated,
                       base_frame_idx=base_idx, rows=rows)


def save_report(res: MergeResult, frame_paths: list[str | Path],
                out_dir: str | Path) -> list[Path]:
    """写 merged.jpg / annotated.jpg / report.csv，返回文件路径列表。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [Path(p).name for p in sorted(Path(p) for p in frame_paths)]
    p_merged = out_dir / "merged.jpg"
    p_annot = out_dir / "annotated.jpg"
    p_csv = out_dir / "report.csv"
    cv2.imwrite(str(p_merged), res.merged, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(p_annot), res.annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(p_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["person_id", "action", "base_frame", "src_frame",
                    "ear_base", "ear_src", "score_base", "score_src"])
        for r in res.rows:
            w.writerow([r.person_id, r.action,
                        names[res.base_frame_idx],
                        names[r.src_frame],
                        f"{r.ear_base:.3f}", f"{r.ear_src:.3f}",
                        f"{r.score_base:.3f}", f"{r.score_src:.3f}"])
    return [p_merged, p_annot, p_csv]
