"""clinaudit —— 临床预测模型可信度审计工具。

本包回答一个临床研究者最该关心、却最常被忽略的问题：

    一个预测模型在论文里报了很高的 AUC，它真的能用于临床决策吗？

仅看 AUC 不足以回答。本包从五个维度做系统审计：

1. **判别力**（discrimination）：AUC、PR-AUC、敏感度/特异度
2. **校准**（calibration）：校准曲线、ICI/E50/E90、重校准
3. **公平性**（fairness）：亚组间性能差异、机会均等、差异性影响
4. **临床效用**（utility）：决策曲线分析、净收益
5. **风险控制**（abstention）：选择性预测、conformal 预测

并输出一份可直接用于报告的中文 HTML 审计报告。
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
