"""公平性指标：模型在不同人群亚组上的表现是否一致。

临床 AI 的一个核心风险是：模型在整体人群上表现良好，但在某个亚组
（老年、女性、某家医院）上系统性失效。这类问题不会体现在总体 AUC 上。

本模块同时支持：
- **离散敏感属性**（性别、中心）：直接分组比较
- **连续属性的分组**（年龄）：先按分位数离散化

报告三类差异
------------
1. **性能差异**：AUC 差、敏感度差、特异度差
2. **校准差异**：各亚组的 ICI 与平均偏差
3. **差异性影响**（disparate impact）：不同亚组的阳性预测率之比
"""

from __future__ import annotations

import numpy as np

from .calibration import calibration_curve_data
from .discrimination import confusion_at_threshold, roc_auc


def make_groups(
    values: np.ndarray,
    n_bins: int = 2,
    labels: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """把连续或离散属性转成分组标签。

    对连续属性用中位数（n_bins=2）或分位数切分；对已是离散值的属性
    直接作为组名。
    """
    v = np.asarray(values)
    uniq = np.unique(v[~_isnan(v)])
    if len(uniq) <= max(n_bins, 2) and len(uniq) <= 5:
        # 视作离散属性
        out = np.array([str(x) for x in v], dtype=object)
        return out, sorted(set(out.tolist()))

    # 连续属性：按分位数切分
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.quantile(v[~_isnan(v)], qs)
    idx = np.digitize(v, edges)
    out = np.array([f"Q{i + 1}" for i in idx], dtype=object)
    if labels:
        out = np.array([labels[int(i)] if int(i) < len(labels) else f"Q{i + 1}" for i in idx], dtype=object)
    return out, sorted(set(out.tolist()))


def _isnan(a: np.ndarray) -> np.ndarray:
    try:
        return np.isnan(a.astype(float))
    except (TypeError, ValueError):
        return np.zeros(len(a), dtype=bool)


def subgroup_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """计算每个亚组的完整指标。"""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    g = np.asarray(groups, dtype=object)

    out: dict[str, dict] = {}
    for name in sorted(set(g.tolist())):
        m = g == name
        if m.sum() < 5:
            out[str(name)] = {"n": int(m.sum()), "note": "样本过少，跳过"}
            continue
        yy, pp = y[m], p[m]
        cm = confusion_at_threshold(yy, pp, threshold) if len(np.unique(yy)) > 1 else {}
        cal = calibration_curve_data(yy, pp, n_bins=5) if len(np.unique(yy)) > 1 else {}
        out[str(name)] = {
            "n": int(m.sum()),
            "prevalence": float(yy.mean()),
            "auc": roc_auc(yy, pp) if len(np.unique(yy)) > 1 else float("nan"),
            "sensitivity": cm.get("sensitivity", float("nan")),
            "specificity": cm.get("specificity", float("nan")),
            "ppv": cm.get("ppv", float("nan")),
            "positive_rate": float((pp >= threshold).mean()),
            "mean_predicted": float(pp.mean()),
            "mean_gap": cal.get("mean_gap", float("nan")),
            "ici": cal.get("ici", float("nan")),
        }
    return out


def fairness_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """汇总公平性差异并给出红旗提示。"""
    per = subgroup_metrics(y_true, y_prob, groups, threshold)
    usable = {k: v for k, v in per.items() if "note" not in v}

    if len(usable) < 2:
        return {
            "groups": per,
            "n_groups": len(usable),
            "note": "有效亚组少于 2 个，无法比较公平性",
            "flags": [],
        }

    aucs = [v["auc"] for v in usable.values() if np.isfinite(v["auc"])]
    sens = [v["sensitivity"] for v in usable.values() if np.isfinite(v["sensitivity"])]
    specs = [v["specificity"] for v in usable.values() if np.isfinite(v["specificity"])]
    ppvs = [v["ppv"] for v in usable.values() if np.isfinite(v["ppv"])]
    rates = [v["positive_rate"] for v in usable.values()]
    gaps = [v["mean_gap"] for v in usable.values() if np.isfinite(v["mean_gap"])]

    diffs = {
        "auc_range": float(max(aucs) - min(aucs)) if len(aucs) > 1 else float("nan"),
        "sensitivity_range": float(max(sens) - min(sens)) if len(sens) > 1 else float("nan"),
        "specificity_range": float(max(specs) - min(specs)) if len(specs) > 1 else float("nan"),
        "ppv_range": float(max(ppvs) - min(ppvs)) if len(ppvs) > 1 else float("nan"),
        "calibration_gap_range": float(max(gaps) - min(gaps)) if len(gaps) > 1 else float("nan"),
        "disparate_impact_ratio": float(min(rates) / max(rates)) if max(rates) > 0 else float("nan"),
    }

    flags = _flags(diffs)
    return {
        "groups": per,
        "n_groups": len(usable),
        "diffs": diffs,
        "flags": flags,
        "threshold": float(threshold),
    }


def _flags(diffs: dict) -> list[dict]:
    """根据差异大小给出红旗提示（阈值参考临床 AI 公平性实践）。"""
    out = []
    a = diffs.get("auc_range")
    if np.isfinite(a) and a > 0.10:
        out.append({"level": "高", "metric": "AUC 组间差异", "value": round(a, 3),
                    "message": "亚组间判别力差异超过 0.10，模型在某亚组上可能不可靠"})
    s = diffs.get("sensitivity_range")
    if np.isfinite(s) and s > 0.15:
        out.append({"level": "高", "metric": "敏感度组间差异", "value": round(s, 3),
                    "message": "亚组间漏诊率差异显著，可能导致某些人群被系统性延误诊断"})
    g = diffs.get("calibration_gap_range")
    if np.isfinite(g) and g > 0.10:
        out.append({"level": "中", "metric": "校准偏差组间差异", "value": round(g, 3),
                    "message": "亚组的风险估计系统性偏差不同，需分亚组重校准"})
    di = diffs.get("disparate_impact_ratio")
    if np.isfinite(di) and di < 0.8:
        out.append({"level": "中", "metric": "差异性影响比", "value": round(di, 3),
                    "message": "不同亚组的干预率之比低于 0.8（四分之五法则），需审查分配公平性"})
    return out
