"""校准指标：模型输出的概率是否可信。

为什么校准比 AUC 更重要
-----------------------
AUC 只关心**排序**。一个把所有风险都系统性高估 20 个百分点的模型，AUC 可以
很高，但在临床上会导致过度治疗。校准衡量的是"预测概率"与"实际发生率"是否一致。

主要指标
--------
- **校准曲线**：按预测概率分箱，比较预测均值与实际阳性率。
- **ICI**（Integrated Calibration Index）：校准曲线与对角线的加权平均绝对偏差，
  按样本密度加权。越小越好。
- **E50 / E90**：预测概率与实际概率之差的 50% / 90% 分位数。
  E50 是中位偏差，E90 反映最差情况。
- **Brier 分数**：均方误差，同时反映判别力与校准，越小越好。
- **校准截距 / 斜率**：把标签对 logit(预测) 做逻辑回归，
  截距反映"整体高估/低估"，斜率反映"过度/不足自信"。
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def _validate(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(float).ravel()
    p = np.asarray(y_prob).astype(float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"长度不一致: y={y.shape}, p={p.shape}")
    if not np.all((p >= 0) & (p <= 1)):
        raise ValueError("概率必须在 [0, 1] 区间")
    return y, p


def calibration_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> dict:
    """计算校准曲线。

    Parameters
    ----------
    strategy
        ``"quantile"`` 按预测概率分位分箱（每箱样本数相近，推荐）；
        ``"uniform"`` 按 [0,1] 等宽分箱。
    """
    y, p = _validate(y_true, y_prob)
    n = len(y)
    if n == 0:
        raise ValueError("空输入")

    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    elif strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    else:
        raise ValueError("strategy 必须是 'quantile' 或 'uniform'")
    if len(edges) < 2:
        edges = np.array([0.0, 1.0])

    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        cnt = int(m.sum())
        if cnt == 0:
            continue
        rows.append(
            {
                "bin": b,
                "n": cnt,
                "mean_predicted": float(p[m].mean()),
                "fraction_positive": float(y[m].mean()),
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
            }
        )

    gaps = np.array([r["fraction_positive"] - r["mean_predicted"] for r in rows])
    weights = np.array([r["n"] for r in rows], dtype=float)
    weights = weights / weights.sum()

    return {
        "bins": rows,
        "n_bins_used": len(rows),
        "ici": float(np.sum(weights * np.abs(gaps))),
        "e50": float(np.quantile(np.abs(gaps), 0.5)) if len(gaps) else float("nan"),
        "e90": float(np.quantile(np.abs(gaps), 0.9)) if len(gaps) else float("nan"),
        "mean_gap": float(np.sum(weights * gaps)),
    }


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier 分数：mean((p - y)^2)，越小越好。"""
    y, p = _validate(y_true, y_prob)
    return float(np.mean((p - y) ** 2))


def brier_decomposition(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Brier 分数的分解：可靠性 + 分辨率 - 不确定性。

    .. math::
        BS = \\underbrace{\\frac{1}{N}\\sum_k n_k (\\bar p_k - \\bar y_k)^2}_{\\text{可靠性（越小越好）}}
           - \\underbrace{\\frac{1}{N}\\sum_k n_k (\\bar y_k - \\bar y)^2}_{\\text{分辨率（越大越好）}}
           + \\underbrace{\\bar y (1-\\bar y)}_{\\text{不确定性}}
    """
    y, p = _validate(y_true, y_prob)
    curve = calibration_curve_data(y, p, n_bins=n_bins)
    n = len(y)
    y_bar = float(y.mean())

    reliability = 0.0
    resolution = 0.0
    for r in curve["bins"]:
        w = r["n"] / n
        reliability += w * (r["mean_predicted"] - r["fraction_positive"]) ** 2
        resolution += w * (r["fraction_positive"] - y_bar) ** 2
    uncertainty = y_bar * (1 - y_bar)

    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "brier": brier_score(y, p),
        "check_sum": float(reliability - resolution + uncertainty),
    }


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """校准截距与斜率。

    把真实标签对 logit(预测概率) 做无正则逻辑回归：

        logit(P(y=1)) = a + b * logit(p)

    - ``b ≈ 1, a ≈ 0``：校准良好
    - ``b < 1``：过度自信（预测过于极端）
    - ``b > 1``：信心不足
    - ``a > 0`` 且 ``b ≈ 1``：系统性低估风险
    """
    y, p = _validate(y_true, y_prob)
    if len(np.unique(y)) < 2:
        return {"intercept": float("nan"), "slope": float("nan"), "note": "标签只有单一类别"}
    z = _logit(p).reshape(-1, 1)
    try:
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(z, y)
        return {"intercept": float(lr.intercept_[0]), "slope": float(lr.coef_[0, 0])}
    except Exception as exc:  # noqa: BLE001
        return {"intercept": float("nan"), "slope": float("nan"), "note": str(exc)}


def net_benefit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict:
    """决策曲线分析（Decision Curve Analysis）的净收益。

    在阈值概率 :math:`p_t` 下，把预测概率 >= :math:`p_t` 的样本判为阳性：

    .. math::
        NB = \\frac{TP}{N} - \\frac{FP}{N} \\cdot \\frac{p_t}{1 - p_t}

    参考线：
    - **全部干预**：把所有样本判为阳性
    - **全部不干预**：NB = 0

    模型只有在某阈值区间内高于这两条参考线，才具有临床价值。
    """
    y, p = _validate(y_true, y_prob)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    th = np.asarray(thresholds, dtype=float)
    n = len(y)
    prev = float(y.mean())

    model_nb, treat_all_nb = [], []
    for t in th:
        pred = p >= t
        tp = float(np.sum(pred & (y == 1)))
        fp = float(np.sum(pred & (y == 0)))
        w = t / (1 - t)
        model_nb.append(tp / n - fp / n * w)
        treat_all_nb.append(prev - (1 - prev) * w)

    return {
        "thresholds": th.tolist(),
        "model": model_nb,
        "treat_all": treat_all_nb,
        "treat_none": [0.0] * len(th),
    }


def calibration_summary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """一次性给出全部校准相关指标。"""
    y, p = _validate(y_true, y_prob)
    curve = calibration_curve_data(y, p, n_bins=n_bins)
    out = {
        "ici": curve["ici"],
        "e50": curve["e50"],
        "e90": curve["e90"],
        "mean_gap": curve["mean_gap"],
        "brier": brier_score(y, p),
        "slope": calibration_slope_intercept(y, p)["slope"],
        "intercept": calibration_slope_intercept(y, p)["intercept"],
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "mean_predicted": float(p.mean()),
    }
    out["calibration_curve"] = curve["bins"]
    return out
