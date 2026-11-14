"""gpmerge —— AI 辅助多人合影：离线多帧优势合并 demo 核心包。

管线：人脸检测/关键点 → 跨帧身份关联 → 逐人质量评分 →
      基准帧选取 → 逐人对齐替换 → 无缝融合 → 成片 + 评分报告
"""

from .pipeline import merge_burst, MergeConfig  # noqa: F401
