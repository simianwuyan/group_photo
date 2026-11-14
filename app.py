"""Gradio 演示界面。

Tab 1 多帧合并：上传连拍多帧 -> 合并成片 + 标注图 + 逐人报告。
Tab 2 合成演示：上传单张合影 -> 合成带闭眼/抖动的连拍 -> 基准帧 vs 成片对比。
（路演用 Tab 2：任何一张合影都能现场演示"闭眼修复"）
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from gpmerge import MergeConfig, merge_burst  # noqa: E402
from gpmerge.capture import capture_burst  # noqa: E402
from gpmerge.pipeline import save_report  # noqa: E402
from gpmerge.synth import make_burst  # noqa: E402

MODEL = ROOT / "models" / "face_landmarker.task"
IS_SPACE = bool(
    os.environ.get("SPACE_ID")                             # HF Spaces
    or os.environ.get("GPMERGE_CLOUD")                     # 显式开关（任意云端托管）
    or any(k.startswith("MODELSCOPE") for k in os.environ)  # 魔搭创空间
)  # 云端没有摄像头设备：隐藏摄像头连拍 Tab
_LM = None  # 进程内复用，首次调用时懒加载

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/latest/face_landmarker.task")


def _ensure_model() -> None:
    """模型缺失时自动下载（HF Space / 新机器克隆后首次运行会走到）。"""
    if MODEL.exists():
        return
    import urllib.request
    MODEL.parent.mkdir(exist_ok=True)
    print(f"[model] 模型缺失，下载 face_landmarker.task -> {MODEL}")
    urllib.request.urlretrieve(MODEL_URL, str(MODEL))
    print("[model] 下载完成")


def _landmarker():
    global _LM
    if _LM is None:
        _ensure_model()
        from gpmerge.landmarks import Landmarker
        _LM = Landmarker(MODEL)
    return _LM


def _to_rgb(img_bgr: cv2.Mat) -> cv2.Mat:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


_REPORT_HEADERS = ["人", "动作", "基准分", "来源分", "EAR基准", "EAR来源"]


def _report_rows(res) -> list[list[str]]:
    """MergeResult -> 逐人报告表（Tab 1 / Tab 3 共用）。"""
    return [["P" + str(r.person_id),
             "保留" if r.action == "kept" else
             f"替换 ← 第{r.src_frame}帧" if r.action == "swapped" else
             f"补回 ← 第{r.src_frame}帧",
             f"{r.score_base:.3f}", f"{r.score_src:.3f}",
             f"{r.ear_base:.3f}", f"{r.ear_src:.3f}"] for r in res.rows]


# ---------- Tab 1：多帧合并 ----------

def run_merge(files: list, min_gain: float) -> tuple:
    if not files:
        raise gr.Error("请先上传连拍帧（3 张以上效果最佳）")
    paths = [Path(f.name if hasattr(f, "name") else f) for f in files]
    res = merge_burst(paths, MODEL, cfg=MergeConfig(min_gain=min_gain),
                      landmarker=_landmarker())
    out_dir = Path(tempfile.mkdtemp(prefix="gpmerge_"))
    save_report(res, paths, out_dir)

    rows = _report_rows(res)
    names = sorted(p.name for p in paths)
    info = (f"基准底片：**{names[res.base_frame_idx]}**（{len(paths)} 帧输入）\n\n"
            f"替换 {sum(1 for r in res.rows if r.action != 'kept')} 人 / "
            f"共 {len(res.rows)} 人")
    return _to_rgb(res.merged), _to_rgb(res.annotated), info, rows


# ---------- Tab 2：合成演示 ----------

def run_demo(img: cv2.Mat, n_frames: int, blink_prob: float, seed: int) -> tuple:
    if img is None:
        raise gr.Error("请先上传一张合影")
    work = Path(tempfile.mkdtemp(prefix="gpsynth_"))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    tmp_in = work / "base.jpg"
    cv2.imwrite(str(tmp_in), bgr)

    gt = make_burst(tmp_in, work / "burst", n_frames=int(n_frames),
                    blink_prob=float(blink_prob), seed=int(seed),
                    landmarker=_landmarker())
    frame_paths = sorted((work / "burst").glob("frame_*.jpg"))
    res = merge_burst(frame_paths, MODEL, landmarker=_landmarker())

    n_blink = sum(len(v) for v in gt["blink_frames"].values())
    n_swap = sum(1 for r in res.rows if r.action != "kept")
    info = (f"合成 {len(frame_paths)} 帧、{n_blink} 处闭眼 → "
            f"基准底片第 {res.base_frame_idx} 帧，替换 {n_swap} 人")
    return _to_rgb(res.merged), _to_rgb(res.annotated), info


# ---------- Tab 3：摄像头连拍（真实数据） ----------

def run_camera(cam_index: float, n_frames: float) -> tuple:
    try:
        frames = capture_burst(int(cam_index), n_frames=int(n_frames))
    except RuntimeError as e:
        raise gr.Error(str(e))

    # 真实连拍就地归档，供 realcheck.py 复盘校准评分权重
    burst_dir = ROOT / "data" / "real_burst" / time.strftime("%Y%m%d_%H%M%S")
    burst_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, f in enumerate(frames):
        p = burst_dir / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(p), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths.append(p)

    try:
        res = merge_burst(paths, MODEL, landmarker=_landmarker())
    except RuntimeError as e:
        raise gr.Error(f"连拍不可用：{e}（请确认镜头前有合影对象）")
    save_report(res, paths, burst_dir)

    info = (f"抓取 {len(paths)} 帧，成片与连拍已归档 `data/real_burst/{burst_dir.name}`\n\n"
            f"基准底片：**{paths[res.base_frame_idx].name}**\n\n"
            f"替换 {sum(1 for r in res.rows if r.action != 'kept')} 人 / "
            f"共 {len(res.rows)} 人")
    return _to_rgb(res.merged), _to_rgb(res.annotated), info, _report_rows(res)


with gr.Blocks(title="AI 多人合影助手 · 离线 Demo") as demo:
    gr.Markdown(
        "# AI 辅助多人合影 —— 多帧优势合并（离线 Demo）\n"
        "连拍多帧 → 逐人评分（睁眼/清晰度/亮度/表情/视线）→ 选基准底片 → "
        "只替换状态不佳者 → 无缝融合。全流程端侧，不上传任何数据。")

    with gr.Tab("多帧合并"):
        with gr.Row():
            with gr.Column():
                files_in = gr.File(label="连拍帧（多选 jpg/png）",
                                   file_count="multiple")
                gain = gr.Slider(0.0, 0.3, value=0.05, step=0.01,
                                 label="替换增益阈值（越大越保守）")
                btn = gr.Button("开始合并", variant="primary")
            with gr.Column():
                info1 = gr.Markdown()
                out1 = gr.Image(label="合并成片")
                out2 = gr.Image(label="标注图（绿=保留 橙=替换）")
        table = gr.Dataframe(headers=_REPORT_HEADERS, label="逐人报告")
        btn.click(run_merge, [files_in, gain], [out1, out2, info1, table])

    with gr.Tab("合成演示（路演用）"):
        gr.Markdown("上传一张合影 → 自动合成\"有人闭眼\"的连拍 → 展示 AI 修复结果。"
                    "左边是修复后成片，可对比标注图中的替换痕迹。")
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(label="合影原图（一张）", type="numpy")
                n_f = gr.Slider(6, 20, value=12, step=1, label="连拍帧数")
                p_b = gr.Slider(0.1, 0.9, value=0.35, step=0.05,
                                label="闭眼概率")
                seed_n = gr.Number(value=7, label="随机种子", precision=0)
                btn2 = gr.Button("生成并合并", variant="primary")
            with gr.Column():
                info2 = gr.Markdown()
                out3 = gr.Image(label="合并成片")
                out4 = gr.Image(label="标注图")
        btn2.click(run_demo, [img_in, n_f, p_b, seed_n], [out3, out4, info2])

    if not IS_SPACE:   # 云端没有摄像头设备，隐藏 Tab 3
        with gr.Tab("摄像头连拍（真实数据）"):
            gr.Markdown("笔记本/外接摄像头对准合影对象，一键抓取约 1.6 秒的连拍 → 直接合并。"
                        "连拍与成片自动归档到 `data/real_burst/<时间戳>/`，"
                        "可用 `tools/realcheck.py` 复盘校准评分权重。")
            with gr.Row():
                with gr.Column():
                    cam_idx = gr.Number(value=0, precision=0,
                                        label="摄像头编号（0=内置，1/2=外接）")
                    n_f3 = gr.Slider(8, 16, value=12, step=1, label="连拍帧数")
                    btn3 = gr.Button("开始连拍并合并", variant="primary")
                with gr.Column():
                    info3 = gr.Markdown()
                    out5 = gr.Image(label="合并成片")
                    out6 = gr.Image(label="标注图（绿=保留 橙=替换）")
            table3 = gr.Dataframe(headers=_REPORT_HEADERS, label="逐人报告")
            btn3.click(run_camera, [cam_idx, n_f3], [out5, out6, info3, table3])

if __name__ == "__main__":
    demo.launch()
