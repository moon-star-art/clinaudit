"""判别力指标：模型区分阳性与阴性样本的能力。"""

from __future__ import annotations

import numpy as np


def _validate(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(y_prob).astype(float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"长度不一致: y={y.shape}, p={p.shape}")
    if not np.all((p >= 0) & (p <= 1)):
        raise ValueError("概率必须在 [0, 1] 区间")
    return y, p


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC 曲线下面积（Mann-Whitney U 统计量）。"""
    y, p = _validate(y_true, y_prob)
    pos = p[y == 1]
    neg = p[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # 使用秩平均处理并列
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(pos.size + neg.size, dtype=float)
    ranks[order] = np.arange(1, pos.size + neg.size + 1)
    allv = np.concatenate([pos, neg])
    uniq, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(uniq.size)
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def roc_curve_points(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """返回 ROC 曲线的 (fpr, tpr) 点，用于绘图。"""
    y, p = _validate(y_true, y_prob)
    order = np.argsort(-p, kind="mergesort")
    y = y[order]
    p_sorted = p[order]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]}

    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    # 只在分数变化处取点，避免重复
    keep = np.r_[np.diff(p_sorted) != 0, True]
    tpr = np.r_[0.0, tps[keep] / n_pos]
    fpr = np.r_[0.0, fps[keep] / n_neg]
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist()}


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """PR 曲线下面积（平均精度）。

    类别不平衡时比 ROC-AUC 更能反映实际表现。
    """
    y, p = _validate(y_true, y_prob)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    y = y[order]
    tps = np.cumsum(y == 1)
    precisions = tps / np.arange(1, len(y) + 1)
    recalls = tps / n_pos
    # 平均精度：Σ (R_n - R_{n-1}) * P_n
    ap = float(np.sum(np.diff(np.r_[0.0, recalls]) * precisions))
    return ap


def confusion_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """给定阈值下的混淆矩阵与派生指标。"""
    y, p = _validate(y_true, y_prob)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "accuracy": float((tp + tn) / max(len(y), 1)),
        "youden_j": float(sens + spec - 1) if np.isfinite(sens) and np.isfinite(spec) else float("nan"),
    }


def best_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """找到 Youden 指数最大的阈值。"""
    y, p = _validate(y_true, y_prob)
    cands = np.unique(np.round(p, 6))
    best = None
    for t in cands:
        m = confusion_at_threshold(y, p, float(t))
        if best is None or (m["youden_j"] > best["youden_j"]):
            best = m
    return best or confusion_at_threshold(y, p, 0.5)


def discrimination_summary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """汇总判别力指标。"""
    y, p = _validate(y_true, y_prob)
    out = {
        "auc": roc_auc(y, p),
        "pr_auc": pr_auc(y, p),
        "n": int(len(y)),
        "n_positive": int((y == 1).sum()),
        "prevalence": float(y.mean()),
    }
    out.update(confusion_at_threshold(y, p, threshold))
    out["roc_curve"] = roc_curve_points(y, p)
    return out
