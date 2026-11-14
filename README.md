# AI 辅助多人合影 —— 多帧优势合并（离线 Demo）

> 赛道：AI + 影像 ｜ 场景：社团聚会 / 毕业照 / 志愿活动等多人集体合影
>
> 痛点：合影总有人闭眼、表情不佳或被遮挡，拍摄者顾不过来，废片率高。

## 这是什么

「拍后多帧优势合并」核心功能的**离线算法验证 Demo**：

```
连拍 10+ 帧 → 人脸检测/478点关键点 → 跨帧身份关联 → 逐人逐帧评分
   → 选全局最优帧做"基准底片" → 只替换状态不佳者 → 对齐 + 颜色迁移 + 羽化融合
   → 一张全员状态在线的成片 + 逐人评分报告
```

设计三原则（也是与"AI 重生成"类产品的差异化）：

1. **基准底片**：背景与姿态 90% 来自原图，只动有问题的脸 —— 无"AI 感"；
2. **最小替换**：只替换不达标的人，替换越少痕迹越小；
3. **可解释**：每个人为什么被换、换成哪一帧、前后分数多少，全部可见。

**全流程端侧处理，不上传任何数据**（人脸是敏感个人信息，这既是合规底线也是产品卖点）。

## 快速开始

```bash
# 1. 环境（Python ≥3.10）
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. 模型（约 3.7MB，mediapipe 官方 face_landmarker）
curl -L -o models/face_landmarker.task ^
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

# 3. 一键体验：用任意合影合成"有人闭眼"的连拍并修复
.venv\Scripts\python.exe cli.py --synth data/candidates/photo-1488521787991-ed7bbaae773c.jpg --dir data/burst --out out

# 4. 或直接合并自己的连拍目录
.venv\Scripts\python.exe cli.py --dir 你的连拍目录 --out out

# 4b. 或用本机摄像头现场连拍（真实数据）
.venv\Scripts\python.exe cli.py --camera 0 --n-frames 12 --out out

# 5. Web 界面（PC 端）
.venv\Scripts\python.exe app.py   # 打开 http://127.0.0.1:7860

# 6. 手机连拍（与 app.py 可同时运行）
.venv\Scripts\python.exe phone.py # 终端打印局域网地址+二维码，手机同 Wi-Fi 扫码即开

# 7. 纯端侧手机网站（无需电脑，算法跑在手机浏览器里）
#    mobile/ 静态站，托管到任意静态空间（魔搭 Static / GitHub Pages）即可
```

Web 界面三个标签页：
- **多帧合并**：上传连拍多帧 → 成片 + 标注图 + 逐人报告表
- **合成演示（路演用）**：上传一张合影 → 自动生成含闭眼的连拍 → 展示修复效果
- **摄像头连拍（真实数据）**：摄像头一键抓帧 → 直接合并；连拍自动归档 `data/real_burst/`

手机连拍：运行 `phone.py` 后手机浏览器（同 Wi-Fi）打开打印的地址，后置摄像头
一键抓帧上传合并；首次访问需在浏览器点「高级 → 继续访问」（自签 HTTPS，
浏览器摄像头权限要求安全上下文）。

## 纯端侧手机网站（mobile/）

算法的 JS 移植版：MediaPipe tasks-vision（478 点关键点，GPU delegate）+ OpenCV.js
（仿射/羽化/LAB 合成）+ 手写 Umeyama 相似变换。**连拍、评分、换脸全部在手机
本地完成，零上传零服务器**；PWA 可安装到桌面，首次联网后离线可用。

- `index.html` 手机端页面（连拍 → 本地合并 → 成片/标注/报告）
- `test.html` 对照验证页：载入合成连拍与 Python gt.json 对照
- `test/compare.mjs` node 对照脚本：`node test/compare.mjs test/ref.json`
  （参考数据由 `tools/export_ref.py` 从 Python 管线导出）
- 部署：任意静态托管（`tools/devserver.py` 为本地开发服务器）

## 目录结构

## 目录结构

```
gpmerge/
  landmarks.py   mediapipe FaceLandmarker 封装（478 点，含虹膜/头姿）
  scoring.py     评分：睁眼(EAR)/清晰度/亮度/表情/视线，几何量驱动零训练
  tracking.py    跨帧身份关联（贪心 IoU + 距离兜底）
  pipeline.py    合并管线：基准底片选取、最小替换、相似变换对齐、
                 LAB 颜色迁移、椭圆羽化融合、报告输出
  synth.py       [测试] 合成连拍：皮肤色闭眼 + 手持抖动 + 亮度扰动，自带 GT
  capture.py     [真实连拍] 本机摄像头连拍采集（预热去欠曝，间隔抓帧）
cli.py           命令行入口
app.py           Gradio 界面
phone.py         手机连拍服务（局域网 HTTPS + 二维码，抓帧上传合并）
phone_page.html  手机端抓帧页面
mobile/          纯端侧手机网站（JS 版算法，零上传零服务器，PWA）
tools/
  pick_base.py   候选基图检测评估
  verify.py      自动验证：EAR 可分性 / 决策与 GT 一致性 / 成片复检
  realcheck.py   真实连拍校准（无 GT）：指标表 / EAR 分布 / 拼图核对身份关联
  export_ref.py  导出 Python 管线参考数据（JS 移植对照）
  devserver.py   开发用静态服务器（.mjs/.wasm MIME + 无头验证上报）
data/candidates/ 测试合影（Unsplash）
out*/            运行输出：merged.jpg / annotated.jpg / report.csv
```

## 验证结果

| 用例 | 结果 |
|---|---|
| 随机种子 42（基准帧无闭眼） | VERIFY PASS：全员 kept，选帧避开废片 |
| 强制每帧都有人闭眼（seed 7） | VERIFY PASS：P1 闭眼 EAR 0.131 → 替换后 0.306，成片复检全员睁眼 |
| 视觉检查 | 融合无缝：无色差、无鬼影、无边界 |

闭眼/睁眼 EAR 分离度：闭眼 ≤0.137，睁眼 ≥0.222，双阈值（0.16/0.20）稳定判定。

## 已知边界与下一步

- **人数**：当前示例图 3 人，依赖单人检测质量；毕业照 30+ 人需要补充评测（`num_faces` 参数已留到 30）。
- **真实连拍**：PC 摄像头连拍已支持（Tab 3 / `cli.py --camera`），手机连拍上传也可用（EXIF 方向自动转正）；权重校准用 `tools/realcheck.py` 摊开决策依据。真人眨眼半程帧、表情差异更大，评分权重仍需更多真实数据调优。
- **遮挡/逆光**：本 Demo 只实现闭眼修复；遮挡检测（拍中指导）是产品差异化的另一半，见方案文档规划。
- **侧面脸/低光**：EAR 阈值依赖正脸假设，大侧脸需先按 yaw 分桶调参。

## 演进路线（对应产品方案）

1. **本周**：真实连拍采集（社团/班级），校准评分权重
2. **第 3-6 周**：Android 端 camera2 连拍 + 拍中实时指导（睁眼/看镜头/拥挤三检测先行）
3. **之后**：逐人手动微调滑条、遮挡补拍合成、二维码分发
