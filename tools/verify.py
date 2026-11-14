"""端到端自动验证。

三项检查：
  A. 合成闭眼的 EAR 可分性（闭眼帧 EAR 显著低于睁眼帧）；
  B. 合并决策与 ground truth 一致（基准帧闭眼者必须被换成睁眼帧）；
  C. 成片复检：重新检测 merged.jpg，人脸数一致且全员 EAR 达标。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpmerge.landmarks import Landmarker  # noqa: E402
from gpmerge.scoring import ear_value  # noqa: E402
from gpmerge.tracking import associate  # noqa: E402

EAR_CLOSED = 0.16   # 低于此判定闭眼
EAR_OPEN = 0.20     # 高于此判定睁眼（要求两档间有间隔）


def _fi(name: str) -> int:
    return int(Path(name).stem.split("_")[1])


def main(burst_dir: Path, out_dir: Path) -> bool:
    gt = json.loads((burst_dir / "gt.json").read_text("utf-8"))
    lm = Landmarker(ROOT / "models" / "face_landmarker.task")

    frames = [cv2.imread(str(p)) for p in sorted(burst_dir.glob("frame_*.jpg"))]
    ff = [lm.detect(f) for f in frames]
    associate(ff)
    ears: dict[int, dict[int, float]] = {}
    for fi, faces in enumerate(ff):
        for f in faces:
            ears.setdefault(f.person_id, {})[fi] = ear_value(f)

    ok = True

    # —— A. EAR 可分性 ——
    for pid_s, bfs in gt["blink_frames"].items():
        pid = int(pid_s)
        closed = [ears[pid][k] for k in bfs if k in ears.get(pid, {})]
        opened = [v for k, v in ears.get(pid, {}).items() if k not in bfs]
        c_max = max(closed) if closed else -1
        o_min = min(opened) if opened else 1
        status = "OK" if (c_max < EAR_CLOSED and o_min > EAR_OPEN) else "!!"
        if status == "!!":
            ok = False
        print(f"[A] P{pid} 闭眼EARmax={c_max:.3f} 睁眼EARmin={o_min:.3f} {status}")

    # —— B. 决策与 GT 交叉核对 ——
    rows = list(csv.DictReader(open(out_dir / "report.csv", encoding="utf-8-sig")))
    for r in rows:
        pid = int(r["person_id"])
        base_fi = _fi(r["base_frame"])
        src_fi = _fi(r["src_frame"])
        blinked = base_fi in gt["blink_frames"][str(pid)]
        if blinked:
            good = (r["action"].startswith("swapped")
                    and src_fi not in gt["blink_frames"][str(pid)])
            print(f"[B] P{pid} 基准帧闭眼 -> {r['action']} (src=frame_{src_fi:03d}) "
                  f"{'OK' if good else '!! 未被替换或来源帧仍闭眼'}")
            ok &= good
        else:
            src_ok = src_fi not in gt["blink_frames"][str(pid)]
            ok &= src_ok
            print(f"[B] P{pid} 基准帧睁眼 -> {r['action']} "
                  f"{'OK' if src_ok else '!! 被换到了闭眼帧'}")

    # —— C. 成片复检 ——
    merged = cv2.imread(str(out_dir / "merged.jpg"))
    mfaces = lm.detect(merged)
    n_ok = len(mfaces) == len(gt["persons"])
    print(f"[C] 成片人脸 {len(mfaces)}/{len(gt['persons'])} "
          f"{'OK' if n_ok else '!!'}")
    ok &= n_ok
    for f in mfaces:
        e = ear_value(f)
        e_ok = e > EAR_CLOSED
        ok &= e_ok
        print(f"[C]   成片 EAR={e:.3f} {'OK' if e_ok else '!! 仍闭眼'}")

    print("VERIFY:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    burst = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "burst"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "out"
    sys.exit(0 if main(burst, out) else 1)
