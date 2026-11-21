"""clinaudit.metrics —— 判别力、校准、公平性、临床效用与拒识指标。"""

from __future__ import annotations

from .abstention import (
    abstention_summary,
    aurc,
    confidence_score,
    conformal_prediction_sets,
    empirical_coverage,
    risk_at_coverage,
    selective_curve,
    split_conformal_quantile,
)
from .calibration import (
    brier_decomposition,
    brier_score,
    calibration_curve_data,
    calibration_slope_intercept,
    calibration_summary,
    net_benefit,
)
from .discrimination import (
    best_youden_threshold,
    confusion_at_threshold,
    discrimination_summary,
    pr_auc,
    roc_auc,
    roc_curve_points,
)
from .fairness import fairness_report, make_groups, subgroup_metrics

__all__ = [
    "abstention_summary",
    "aurc",
    "best_youden_threshold",
    "brier_decomposition",
    "brier_score",
    "calibration_curve_data",
    "calibration_slope_intercept",
    "calibration_summary",
    "confidence_score",
    "confusion_at_threshold",
    "conformal_prediction_sets",
    "discrimination_summary",
    "empirical_coverage",
    "fairness_report",
    "make_groups",
    "net_benefit",
    "pr_auc",
    "risk_at_coverage",
    "roc_auc",
    "roc_curve_points",
    "selective_curve",
    "split_conformal_quantile",
    "subgroup_metrics",
]
