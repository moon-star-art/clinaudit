"""拒识与不确定性：模型在什么情况下应当"不表态"。

临床 AI 落地时的一个重要设计是**选择性预测**：模型只对高置信度的样本给出
结论，把不确定的样本交给医生。这比强行给出所有样本的预测更安全。

两个互补的工具
--------------
1. **选择性预测曲线**：覆盖率 vs 风险。按置信度从高到低纳入样本，
   观察错误率如何变化，用于选择部署时的覆盖阈值。
2. **Split Conformal 预测**：给定显著性水平 α，构造预测集合，
   并**在有限样本下严格保证**覆盖率 ≥ 1-α。
"""

from __future__ import annotations

import numpy as np


def _validate(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(y_prob).astype(float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"长度不一致: y={y.shape}, p={p.shape}")
    return y, p


# ------------------------------------------------------------ 选择性预测
def confidence_score(y_prob: np.ndarray) -> np.ndarray:
    """置信度：越接近 0 或 1 越自信。

    定义 ``conf = max(p, 1-p) = 0.5 + |p - 0.5|``，取值 [0.5, 1]。
    """
    p = np.asarray(y_prob, dtype=float).ravel()
    return 0.5 + np.abs(p - 0.5)


def selective_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_points: int = 50,
) -> dict:
    """覆盖率—风险曲线。

    Returns
    -------
    dict 含 ``coverage``、``risk``（在已纳入样本上的错误率）、``n_kept``。
    """
    y, p = _validate(y_true, y_prob)
    n = len(y)
    conf = confidence_score(p)
    order = np.argsort(-conf, kind="mergesort")
    pred = (p >= threshold).astype(int)
    err = (pred != y).astype(float)

    ks = np.unique(np.linspace(1, n, n_points).astype(int))
    coverage, risk, kept = [], [], []
    for k in ks:
        idx = order[:k]
        coverage.append(float(k / n))
        risk.append(float(err[idx].mean()))
        kept.append(int(k))

    return {
        "coverage": coverage,
        "risk": risk,
        "n_kept": kept,
        "base_risk": float(err.mean()),
        "base_coverage": 1.0,
    }


def aurc(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """AURC（Area Under Risk-Coverage curve），越小越好。

    与随机弃权相比，好的置信度估计应显著降低 AURC。
    """
    curve = selective_curve(y_true, y_prob, threshold=threshold, n_points=200)
    cov = np.asarray(curve["coverage"])
    risk = np.asarray(curve["risk"])
    trap = getattr(np, "trapezoid", None) or np.trapz
    # 归一化到 [0,1] 区间面积（覆盖率为横轴）
    area = float(trap(risk, cov))
    span = cov.max() - cov.min()
    return area / span if span > 0 else float("nan")


def risk_at_coverage(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_coverage: float = 0.8,
    threshold: float = 0.5,
) -> dict:
    """达到指定覆盖率时的风险，以及对应的置信度阈值。"""
    if not 0 < target_coverage <= 1:
        raise ValueError("target_coverage 必须在 (0, 1]")
    y, p = _validate(y_true, y_prob)
    n = len(y)
    conf = confidence_score(p)
    order = np.argsort(-conf, kind="mergesort")
    k = max(1, int(round(target_coverage * n)))
    idx = order[:k]
    pred = (p[idx] >= threshold).astype(int)
    err = (pred != y[idx]).astype(float)
    cut = float(conf[order[k - 1]])
    return {
        "target_coverage": float(target_coverage),
        "achieved_coverage": float(k / n),
        "risk": float(err.mean()),
        "confidence_cutoff": cut,
        "prob_band": [max(0.0, 0.5 - (cut - 0.5)), min(1.0, 0.5 + (cut - 0.5))],
        "n_kept": int(k),
    }


# ------------------------------------------------------------ Conformal
def split_conformal_quantile(
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """计算 Split Conformal 的分数分位数。

    使用非一致性分数 ``s = 1 - p(y_true)``，取第
    ``ceil((n+1)(1-alpha))/n`` 分位数。这一修正（+1）保证了有限样本下的
    覆盖率下界，是 conformal 理论的核心。
    """
    y, p = _validate(y_cal, p_cal)
    n = len(y)
    if n < 2:
        raise ValueError("校准集过小")
    p_true = np.where(y == 1, p, 1 - p)
    scores = 1.0 - p_true
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    return float(np.sort(scores)[k - 1])


def conformal_prediction_sets(
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    p_test: np.ndarray,
    alpha: float = 0.1,
) -> dict:
    """为测试样本构造预测集合，并评估经验覆盖率。

    原理
    ----
    非一致性分数取 ``s(y) = 1 - p(y)``。对候选标签 y 逐一判断它是否
    进入集合：

        y 入集合  ⟺  s(y) = 1 - p(y) <= q

    其中 ``q`` 是校准集分数的 ``ceil((n+1)(1-alpha))/n`` 分位数。

    于是：
    - 标签 1：``1 - p1 <= q``  ⟺  ``p1 >= 1 - q``
    - 标签 0：``1 - (1-p1) <= q``  ⟺  ``p1 <= q``

    **两个条件必须分别判断**。只判断"是否包含阳性"会使得集合大小被限制在
    {1} 或 {1,0}，从而丢失 {0} 这一情形，覆盖率会显著低于 1-alpha。
    """
    q = split_conformal_quantile(y_cal, p_cal, alpha=alpha)
    p_test = np.asarray(p_test, dtype=float).ravel()
    include_pos = p_test >= (1 - q)
    include_neg = p_test <= q
    size = include_pos.astype(int) + include_neg.astype(int)
    return {
        "q": float(q),
        "alpha": float(alpha),
        "target_coverage": 1 - alpha,
        "prob_threshold": float(1 - q),
        "prob_upper": float(q),
        "include_positive": include_pos,
        "include_negative": include_neg,
        "predict_positive": include_pos.astype(int),
        "set_size": size.tolist(),
        "mean_set_size": float(size.mean()),
        "empty_set_fraction": float((size == 0).mean()),
    }


def empirical_coverage(y_true: np.ndarray, sets: dict) -> float:
    """预测集合包含真实标签的经验覆盖率。

    对每个样本，检查其真实标签是否在预测集合中：
    - 真实为阳性：要求 ``include_positive`` 为真
    - 真实为阴性：要求 ``include_negative`` 为真

    若某个样本两个标签都不在集合中（空集），该样本必然未被覆盖。
    """
    y = np.asarray(y_true).astype(int).ravel()
    inc_pos = np.asarray(sets["include_positive"]).astype(bool)
    inc_neg = np.asarray(sets["include_negative"]).astype(bool)
    covered = np.where(y == 1, inc_pos, inc_neg)
    return float(covered.mean())


def abstention_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    coverages: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9, 1.0),
) -> dict:
    """汇总拒识相关指标，供报告使用。"""
    y, p = _validate(y_true, y_prob)
    out = {
        "base_risk": float(((p >= threshold).astype(int) != y).mean()),
        "aurc": aurc(y, p, threshold),
        "curve": selective_curve(y, p, threshold),
        "at_coverage": [risk_at_coverage(y, p, c, threshold) for c in coverages],
    }
    return out
