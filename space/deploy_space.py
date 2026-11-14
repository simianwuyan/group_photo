"""一键发布 HF Space：创建 Gradio Space 并上传所需文件。

用法（token 在 HF 网页 Settings -> Access Tokens 创建，勾选 write）：
  .venv\\Scripts\\python.exe space\\deploy_space.py --repo <你的用户名>/<space名> --token hf_xxx

流程：把 Space 所需文件（space/ 的 README+requirements + 根目录的 app.py、
gpmerge/、models/）整理成临时布局目录，然后创建 Space 并整体上传。
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GP_FILES = ["__init__.py", "capture.py", "landmarks.py",
            "pipeline.py", "scoring.py", "synth.py", "tracking.py"]


def stage_dir(tmp: Path) -> None:
    """按 Space 仓库根布局整理文件。"""
    (tmp / "gpmerge").mkdir(parents=True)
    (tmp / "models").mkdir(parents=True)
    shutil.copy2(ROOT / "space" / "README.md", tmp / "README.md")
    shutil.copy2(ROOT / "space" / "requirements.txt", tmp / "requirements.txt")
    shutil.copy2(ROOT / "app.py", tmp / "app.py")
    for name in GP_FILES:
        shutil.copy2(ROOT / "gpmerge" / name, tmp / "gpmerge" / name)
    shutil.copy2(ROOT / "models" / "face_landmarker.task",
                 tmp / "models" / "face_landmarker.task")


def main() -> None:
    ap = argparse.ArgumentParser(description="发布 HF Space")
    ap.add_argument("--repo", required=True, help="如 yourname/group-photo-demo")
    ap.add_argument("--token", required=True, help="HF Access Token（write 权限）")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    tmp = Path(tempfile.mkdtemp(prefix="space_stage_"))
    stage_dir(tmp)
    api = HfApi(token=args.token)
    url = api.create_repo(args.repo, repo_type="space", space_sdk="gradio",
                          private=False, exist_ok=True)
    print("Space:", url)
    api.upload_folder(folder_path=str(tmp), repo_id=args.repo, repo_type="space",
                      commit_message="initial: multi-frame merge demo")
    print("已上传，构建约需 1-3 分钟，完成后访问：", url)


if __name__ == "__main__":
    main()
