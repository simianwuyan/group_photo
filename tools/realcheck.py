"""真实连拍校准工具（无 GT）：指标表 + EAR 分布 + 连拍拼图 + 合并验证。

用法：
  .venv\\Scripts\\python.exe tools\\realcheck.py <连拍目录> [--out 输出目录]

真实连拍没有 gt.json，无法像 tools/verify.py 那样自动判对错；本工具把
决策依据全部摊开供人工校准：
  metrics.csv        逐人逐帧原始指标与总分（评分权重校准的数据基础）
  终端打印           每人 EAR 分布 + 疑似闭眼帧 -> 校准 EAR_BLINK_ABS / EAR_RATIO_OPEN
  contact_sheet.jpg  连拍拼图（人框 + PID 标注）-> 核对跨帧身份关联是否张冠李戴
  merged/annotated/report.csv  合并产物（同 cli.py）
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpmerge import merge_burst  # noqa: E402
from gpmerge.landmarks import DEFAULT_MODEL_PATH, Landmarker  # noqa: E402
from gpmerge.pipeline import _face_gray, imread_exif, save_report  # noqa: E402
from gpmerge.scoring import (brightness_score, ear_value, eyes_open_score,  # noqa: E402
                             gaze_score, sharpness_score, smile_score,
                             total_score)
from gpmerge.tracking import associate  # noqa: E402


def frame_paths(d: Path) -> list[Path]:
    # set 去重：Windows 文件系统 glob 大小写不敏感，*.jpg 与 *.JPG 会重复匹配
    paths = {p for pat in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
             for p in d.glob(pat)}
    return sorted(paths)


def _contact_sheet(frames: list[np.ndarray], ff: list[list],
                   out_path: Path, thumb_h: int = 360, cols: int = 4) -> None:
    """连拍拼图：每帧缩略图上画人框与 P 编号，供肉眼核对身份关联。"""
    thumbs = []
    for fi, (img, faces) in enumerate(zip(frames, ff)):
        s = thumb_h / img.shape[0]
        th = cv2.resize(img, (max(1, int(img.shape[1] * s)), thumb_h))
        for face in faces:
            x, y, w, h = face.box
            cv2.rectangle(th, (int(x * s), int(y * s)),
                          (int((x + w) * s), int((y + h) * s)), (80, 190, 80), 2)
            cv2.putText(th, f"f{fi:02d} P{face.person_id}",
                        (int(x * s), max(16, int(y * s) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 140, 255), 1, cv2.LINE_AA)
        thumbs.append(th)

    rows = []
    for i in range(0, len(thumbs), cols):
        row = thumbs[i:i + cols]
        w = max(t.shape[1] for t in row)
        padded = [np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0)),
                         constant_values=30) for t in row]
        rows.append(np.hstack(padded))
    w = max(r.shape[1] for r in rows)
    padded = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)),
                     constant_values=30) for r in rows]
    cv2.imwrite(str(out_path), np.vstack(padded), [cv2.IMWRITE_JPEG_QUALITY, 92])


def main(burst_dir: Path, out_dir: Path) -> None:
    paths = frame_paths(burst_dir)
    assert paths, f"{burst_dir} 里没有图片"
    frames = [imread_exif(p) for p in paths]
    assert all(f is not None for f in frames), "有读图失败（损坏文件？）"

    lm = Landmarker(DEFAULT_MODEL_PATH)
    ff = [lm.detect(f) for f in frames]
    for i, faces in enumerate(ff):
        for face in faces:
            face.frame_idx = i
    associate(ff)
    person_ids = sorted({f.person_id for faces in ff
                         for f in faces if f.person_id is not None})
    assert person_ids, "连拍中没有检测到人脸"

    # —— 1. 逐人逐帧指标 ——
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    rows: list[dict] = []
    ears: dict[int, dict[int, float]] = {}
    for fi, faces in enumerate(ff):
        for face in faces:
            pid = face.person_id
            g = _face_gray(grays[fi], face)
            ear = ear_value(face)
            ears.setdefault(pid, {})[fi] = ear
            rows.append({"person_id": pid, "frame": fi,
                         "frame_name": paths[fi].name,
                         "ear": ear,
                         "sharp": sharpness_score(g),
                         "bright": brightness_score(g),
                         "smile": smile_score(face),
                         "gaze": gaze_score(face)})

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["person_id", "frame", "frame_name",
                    "ear", "eyes_open", "sharp", "bright", "smile", "gaze", "total"])
        for r in rows:
            max_ear = max(ears[r["person_id"]].values())
            eyes = eyes_open_score(r["ear"], max_ear)
            r["eyes"] = eyes
            r["total"] = total_score(eyes=eyes, sharp=r["sharp"],
                                     bright=r["bright"], smile=r["smile"],
                                     gaze=r["gaze"])
            w.writerow([r["person_id"], r["frame"], r["frame_name"]]
                       + [f"{r[k]:.3f}" for k in
                          ("ear", "eyes", "sharp", "bright", "smile", "gaze", "total")])
    print(f"[metrics] {len(rows)} 条（{len(person_ids)} 人 x {len(paths)} 帧）"
          f" -> {out_dir / 'metrics.csv'}")

    # —— 2. EAR 分布 + 疑似闭眼帧（与管线同款双阈值判定）——
    for pid in person_ids:
        per = ears[pid]
        vals = sorted(per.values())
        max_ear = vals[-1]
        closed = [fi for fi, e in sorted(per.items())
                  if eyes_open_score(e, max_ear) < 0.72 and e < 0.20]
        med = vals[len(vals) // 2]
        print(f"[EAR] P{pid} min={vals[0]:.3f} med={med:.3f} max={max_ear:.3f}"
              + (f"，疑似闭眼帧: {['f%02d' % i for i in closed]}（EAR "
                 f"{'/'.join(f'{per[i]:.3f}' for i in closed)}）" if closed else "，无闭眼"))
    print("[EAR] 若判定与实际不符：调 scoring.py 的 EAR_BLINK_ABS / EAR_RATIO_OPEN")

    # —— 3. 拼图（核对身份关联）——
    sheet = out_dir / "contact_sheet.jpg"
    _contact_sheet(frames, ff, sheet)
    print(f"[sheet] 连拍拼图 -> {sheet}（核对每帧 P 编号是否指向同一个人）")

    # —— 4. 合并产物 ——
    res = merge_burst(paths, landmarker=lm)
    files = save_report(res, paths, out_dir)
    names = [p.name for p in paths]
    print(f"[merge] 基准底片: {names[res.base_frame_idx]}")
    for r in res.rows:
        print(f"  P{r.person_id:<2} {r.action:<16} "
              f"score {r.score_base:.3f}->{r.score_src:.3f}")
    print("[saved]", ", ".join(str(f) for f in files))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="真实连拍校准（无 GT）")
    ap.add_argument("burst_dir", help="连拍帧目录（手机连拍/摄像头连拍均可）")
    ap.add_argument("--out", default=None, help="输出目录（默认 out/realcheck_<目录名>）")
    args = ap.parse_args()
    burst = Path(args.burst_dir)
    out = Path(args.out) if args.out else ROOT / "out" / f"realcheck_{burst.name}"
    main(burst, out)
