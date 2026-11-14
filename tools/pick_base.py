"""评估候选基图：人脸数量、尺寸、检测稳定性，选出适合做 demo 的合影。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpmerge.landmarks import Landmarker  # noqa: E402


def main(cand_dir: Path) -> None:
    lm = Landmarker(ROOT / "models" / "face_landmarker.task")
    rows = []
    for p in sorted(cand_dir.glob("*.jpg")):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        if max(h, w) > 2048:
            s = 2048 / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)))
        faces = lm.detect(img)
        widths = sorted(f.face_width for f in faces)
        med = widths[len(widths) // 2] if widths else 0
        rows.append((p.name, len(faces), med, f"{w}x{h}"))

    print(f"{'file':<44} {'faces':>5} {'median_w':>8}  size")
    for name, n, med, size in rows:
        print(f"{name:<44} {n:>5} {med:>8.0f}  {size}")


if __name__ == "__main__":
    main(ROOT / "data" / "candidates")
