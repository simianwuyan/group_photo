"""导出 JS 移植对照所需的参考数据（在 data/burst 上跑一遍 Python 管线）。

用法：python tools/export_ref.py [burst目录] [--out ref.json]
输出 ref.json：
  framesFaces   每帧每人 pts(478×2)/box/headYaw/Pitch/personId（tracking 与几何对照）
  raw/scores    Python 的逐人逐帧指标与总分（scoring 对照）
  rows/base_idx 合并决策（决策流对照）
  swap_data     每次 swap 的锚点输入与 Python LMEDS 得到的 M（umeyama 对照）
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from gpmerge.landmarks import ALIGN_ANCHORS, Landmarker  # noqa: E402
from gpmerge.pipeline import _frame_shift, _swap_face, merge_burst  # noqa: E402
from gpmerge.tracking import associate  # noqa: E402


def face_to_json(face):
    return {
        "pts": [[round(float(x), 8), round(float(y), 8)] for x, y in face.pts],
        "box": list(face.box),
        "headYaw": face.head_yaw,
        "headPitch": face.head_pitch,
        "personId": face.person_id,
    }


def main(burst_dir: Path, out_path: Path) -> None:
    lm = Landmarker(ROOT / "models" / "face_landmarker.task")
    paths = sorted(burst_dir.glob("frame_*.jpg"))
    frames = [cv2.imread(str(p)) for p in paths]

    # 检测 + 关联（与管线一致）
    ff = [lm.detect(f) for f in frames]
    for i, faces in enumerate(ff):
        for face in faces:
            face.frame_idx = i
    associate(ff)

    # 每帧每人的人脸 ROI 灰度（sharp/bright 对照用）
    from gpmerge.pipeline import _face_gray
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    rois = {}
    for i, faces in enumerate(ff):
        for face in faces:
            roi = _face_gray(grays[i], face)
            rois[f"{face.person_id}_{i}"] = {
                "w": int(roi.shape[1]), "h": int(roi.shape[0]),
                "data": roi.reshape(-1).tolist(),
            }

    # Python 管线（拿到 raw/scores/rows）
    from gpmerge.pipeline import merge_burst as _mb
    res = _mb(paths, landmarker=lm)

    frames_faces = [[face_to_json(f) for f in faces] for faces in ff]

    ref = {
        "names": [p.name for p in paths],
        "framesFaces": frames_faces,
        "raw": {str(pid): {str(fi): m for fi, m in per.items()}
                for pid, per in res.raw.items()},
        "scores": {str(pid): {str(fi): s for fi, s in per.items()}
                   for pid, per in res.scores.items()},
        "base_idx": res.base_frame_idx,
        "rows": [{"personId": r.person_id, "baseFrame": r.base_frame,
                  "srcFrame": r.src_frame, "action": r.action,
                  "earBase": r.ear_base, "earSrc": r.ear_src,
                  "scoreBase": r.score_base, "scoreSrc": r.score_src}
                 for r in res.rows],
    }

    # swap 锚点与 Python M（重放决策路径采集）
    swap_data = []
    frames_faces_obj = ff
    base_idx = res.base_frame_idx
    cfg_like = res
    for r in res.rows:
        if r.action == "kept":
            continue
        pid, src_fi = r.person_id, r.src_frame
        dst_face = next((f for f in frames_faces_obj[base_idx] if f.person_id == pid), None)
        src_face = next(f for f in frames_faces_obj[src_fi] if f.person_id == pid)
        if dst_face is None:
            M = _frame_shift(frames_faces_obj, base_idx, src_fi)
            swap_data.append({"pid": pid, "kind": "frame_shift", "srcFi": src_fi,
                              "M": None if M is None else [[float(v) for v in row] for row in M]})
        else:
            sp = src_face.pts[list(ALIGN_ANCHORS)].astype(np.float32)
            dp = dst_face.pts[list(ALIGN_ANCHORS)].astype(np.float32)
            import cv2 as _cv2
            M, _ = _cv2.estimateAffinePartial2D(sp, dp, method=_cv2.LMEDS)
            swap_data.append({"pid": pid, "kind": "face_swap", "srcFi": src_fi,
                              "sp": [[float(x), float(y)] for x, y in sp],
                              "dp": [[float(x), float(y)] for x, y in dp],
                              "M": None if M is None else [[float(v) for v in row] for row in M]})
    ref["swap_data"] = swap_data
    ref["rois"] = rois

    out_path.write_text(json.dumps(ref), "utf-8")
    print(f"ref.json 写出: {out_path}（{out_path.stat().st_size} 字节）")


if __name__ == "__main__":
    burst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "burst"
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else ROOT / "mobile" / "test" / "ref.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    main(burst, out)
