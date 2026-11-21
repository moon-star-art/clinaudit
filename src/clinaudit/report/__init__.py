"""clinaudit.report —— 中文 HTML 审计报告与图表。"""

from __future__ import annotations

from .render import (
    plot_calibration,
    plot_decision_curve,
    plot_roc,
    plot_selective,
    plot_subgroup_bars,
    render_report,
)

__all__ = [
    "plot_calibration",
    "plot_decision_curve",
    "plot_roc",
    "plot_selective",
    "plot_subgroup_bars",
    "render_report",
]
