"""ModelScope 创空间引导文件：启动时解包 ms.zip，再运行真正的应用。

背景：创空间网页上传 zip 只原样存储、不自动解压，且构建需要根目录
app.py 与 requirements.txt。故仓库根只放 3 个文件：
  app.py          本引导文件（上传后会被 zip 内的真正 app.py 覆盖）
  requirements.txt  依赖清单（构建阶段安装）
  ms.zip          真正的应用（app.py / gpmerge/ / models/）
冷启动时：解包 ms.zip -> 覆盖本文件为真正的 app.py -> 以 __main__ 执行。
热重启时：app.py 已是真正的应用，ms.zip 已改名 ms.zip.loaded -> 直接运行。
"""

import runpy
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZIP = HERE / "ms.zip"

if ZIP.exists():
    print("[bootstrap] 解包 ms.zip ->", HERE)
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(HERE)
    ZIP.rename(HERE / "ms.zip.loaded")   # 标记已解包（重启后仓库重置则重复解包）

sys.path.insert(0, str(HERE))
runpy.run_path(str(HERE / "app.py"), run_name="__main__")
