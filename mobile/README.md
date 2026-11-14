# mobile/ —— 纯端侧手机网站

算法的 JS 移植版：**连拍、逐人评分、跨帧关联、换脸合成全部在手机浏览器
本地完成，零上传零服务器**（GitHub Pages / 魔搭 Static 部署后即可使用）。

- `index.html` 手机端页面：开启后置摄像头 → 一键连拍 → 本地合并 → 成片/标注图/逐人报告（PWA 可安装桌面，首次联网后离线可用）
- `test.html` 对照验证页：载入合成连拍，与 Python 管线的 gt.json 对照（`?autorun=1` 自动运行）
- `test/compare.mjs` node 对照脚本：`node test/compare.mjs test/ref.json`
  （参考数据由 `tools/export_ref.py` 从 Python 管线导出到 `test/ref*.json`，不入库）
- `vendor/` 固定版本的 opencv.js 与 tasks-vision（wasm 内嵌，无 CDN 依赖）
- `models/face_landmarker.task` 与 Python 端共用同一模型

算法参数与 `gpmerge/`（Python 版）逐字对齐：EAR 双阈值 0.14/0.72、五维权重
(0.45, 0.20, 0.10, 0.15, 0.10)、最小替换增益 0.05、椭圆羽化 0.12。
