"""命令行入口。

用法：
  python cli.py --dir data/burst --out out
  python cli.py --camera 0 --n-frames 12 --out out    # 本机摄像头真实连拍
  python cli.py --synth data/candidates/xxx.jpg --dir data/burst   # 先合成连拍再合并
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from gpmerge import MergeConfig, merge_burst  # noqa: E402
from gpmerge.capture import capture_burst  # noqa: E402
from gpmerge.pipeline import save_report  # noqa: E402
from gpmerge.synth import make_burst  # noqa: E402

MODEL = ROOT / "models" / "face_landmarker.task"


def frame_paths(d: Path) -> list[Path]:
    # set 去重：Windows 文件系统 glob 大小写不敏感，*.jpg 与 *.JPG 会重复匹配
    paths = {p for pat in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
             for p in d.glob(pat)}
    return sorted(paths)


def main() -> None:
    ap = argparse.ArgumentParser(description="AI 多帧合影合并 demo")
    ap.add_argument("--dir", help="连拍帧目录")
    ap.add_argument("--camera", metavar="N", type=int,
                    help="[真实连拍] 用本机摄像头抓帧再合并（与 --dir 二选一）")
    ap.add_argument("--out", default="out", help="输出目录")
    ap.add_argument("--synth", metavar="IMG",
                    help="[测试] 用一张合影合成带闭眼/抖动的连拍再合并（需 --dir）")
    ap.add_argument("--n-frames", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.camera is None and not args.dir:
        ap.error("需要 --dir <连拍目录> 或 --camera <编号> 之一")
    if args.synth and args.camera is not None:
        ap.error("--synth 不能与 --camera 同时使用")

    if args.camera is not None:
        burst_dir = Path(args.out) / f"real_burst_{time.strftime('%Y%m%d_%H%M%S')}"
        print(f"[camera] 摄像头 {args.camera} 连拍 {args.n_frames} 帧 -> {burst_dir}")
        frames = capture_burst(args.camera, n_frames=args.n_frames)
        burst_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            cv2.imwrite(str(burst_dir / f"frame_{i:03d}.jpg"), f,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        src = burst_dir
    else:
        src = Path(args.dir)

    from gpmerge.landmarks import Landmarker
    lm = Landmarker(MODEL)   # 进程内单实例（同进程第二个 FaceLandmarker 初始化会死锁）

    if args.synth:
        gt = make_burst(args.synth, src, n_frames=args.n_frames, seed=args.seed,
                        landmarker=lm)
        n_blinks = sum(len(v) for v in gt["blink_frames"].values())
        print(f"[synth] {len(gt['persons'])} 人 x {args.n_frames} 帧，"
              f"共 {n_blinks} 处闭眼 -> {src}")

    paths = frame_paths(src)
    assert paths, f"{src} 里没有图片"
    res = merge_burst(paths, MODEL, landmarker=lm)

    names = [p.name for p in paths]
    print(f"[merge] 基准底片: {names[res.base_frame_idx]}")
    for r in res.rows:
        ear = f" ear {r.ear_base:.3f}->{r.ear_src:.3f}" if r.action.startswith("swapped") else ""
        print(f"  P{r.person_id:<2} {r.action:<16} "
              f"score {r.score_base:.3f}->{r.score_src:.3f}{ear}")

    files = save_report(res, paths, args.out)
    print("[saved]", ", ".join(str(f) for f in files))


if __name__ == "__main__":
    main()
