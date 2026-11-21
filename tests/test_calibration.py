"""校准指标测试。

校准是本项目的核心，测试重点：
- 完美校准的合成数据应得到接近 0 的 ICI
- 系统性高估/低估应被正确识别
- Brier 分解各项之和应等于 Brier 分数
"""

from __future__ import annotations

import numpy as np
import pytest

from clinaudit.metrics.calibration import (
    brier_decomposition,
    brier_score,
    calibration_curve_data,
    calibration_slope_intercept,
    calibration_summary,
    net_benefit,
)


def perfectly_calibrated(n: int = 20000, seed: int = 0):
    """构造严格校准的数据：p ~ U(0,1)，y ~ Bernoulli(p)。"""
    rng = np.random.default_rng(seed)
    p = rng.random(n)
    y = (rng.random(n) < p).astype(int)
    return y, p


def systematically_overestimated(n: int = 5000, shift: float = 0.2):
    """构造系统性高估：真实概率比预测低 shift。"""
    rng = np.random.default_rng(1)
    p = rng.uniform(shift, 1.0, n)
    true_p = p - shift
    y = (rng.random(n) < true_p).astype(int)
    return y, p


def test_calibration_curve_bins_valid():
    y, p = perfectly_calibrated(2000)
    c = calibration_curve_data(y, p, n_bins=10)
    assert c["n_bins_used"] > 1
    assert all(b["n"] > 0 for b in c["bins"])
    assert sum(b["n"] for b in c["bins"]) == len(y)


def test_ici_near_zero_for_calibrated():
    y, p = perfectly_calibrated()
    c = calibration_curve_data(y, p, n_bins=10)
    assert c["ici"] < 0.02


def test_ici_large_for_miscalibrated():
    y, p = systematically_overestimated()
    c = calibration_curve_data(y, p, n_bins=10)
    assert c["ici"] > 0.15


def test_mean_gap_sign_for_overestimation():
    """高估时 mean_gap（实际-预测）应为负。"""
    y, p = systematically_overestimated()
    c = calibration_curve_data(y, p, n_bins=10)
    assert c["mean_gap"] < -0.1


def test_calibration_curve_uniform_strategy():
    y, p = perfectly_calibrated(3000)
    c = calibration_curve_data(y, p, n_bins=8, strategy="uniform")
    assert c["n_bins_used"] <= 8


def test_calibration_curve_bad_strategy():
    y, p = perfectly_calibrated(200)
    with pytest.raises(ValueError, match="strategy"):
        calibration_curve_data(y, p, strategy="random")


def test_calibration_curve_length_mismatch():
    with pytest.raises(ValueError, match="长度不一致"):
        calibration_curve_data(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))


def test_calibration_curve_invalid_probability():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        calibration_curve_data(np.array([0, 1]), np.array([0.1, 1.5]))


def test_brier_perfect_prediction():
    y = np.array([1, 1, 0, 0])
    p = np.array([1.0, 1.0, 0.0, 0.0])
    assert brier_score(y, p) == 0.0


def test_brier_worst_prediction():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert brier_score(y, p) == 1.0


def test_brier_known_value():
    y = np.array([1, 0])
    p = np.array([0.9, 0.2])
    expected = ((0.9 - 1) ** 2 + (0.2 - 0) ** 2) / 2
    assert np.isclose(brier_score(y, p), expected)


def test_brier_decomposition_sums_correctly():
    """可靠性 - 分辨率 + 不确定性 ≈ Brier 分数。

    注意这是**近似**等式：分解按箱内的平均预测概率计算，只有当每箱内
    预测值完全相同时才严格相等。分箱越细误差越小，本测试用 10 箱，
    容差取 5e-3。
    """
    y, p = perfectly_calibrated(3000)
    d = brier_decomposition(y, p, n_bins=10)
    assert np.isclose(d["check_sum"], d["brier"], atol=5e-3)


def test_brier_decomposition_error_shrinks_with_more_bins():
    """分箱越细，分解与 Brier 的偏差应越小。"""
    y, p = perfectly_calibrated(5000)
    err10 = abs(brier_decomposition(y, p, n_bins=10)["check_sum"] - brier_score(y, p))
    err50 = abs(brier_decomposition(y, p, n_bins=50)["check_sum"] - brier_score(y, p))
    assert err50 <= err10 + 1e-9


def test_brier_reliability_small_when_calibrated():
    y, p = perfectly_calibrated()
    d = brier_decomposition(y, p, n_bins=10)
    assert d["reliability"] < 0.005


def test_calibration_slope_near_one_when_calibrated():
    y, p = perfectly_calibrated()
    s = calibration_slope_intercept(y, p)
    assert 0.9 < s["slope"] < 1.1
    assert abs(s["intercept"]) < 0.1


def test_calibration_slope_below_one_when_overconfident():
    """过度自信（预测过于极端）时斜率应 < 1。"""
    rng = np.random.default_rng(3)
    n = 8000
    true_p = rng.uniform(0.3, 0.7, n)
    # 预测被推向极端
    p = np.clip(0.5 + (true_p - 0.5) * 2.5, 0.01, 0.99)
    y = (rng.random(n) < true_p).astype(int)
    s = calibration_slope_intercept(y, p)
    assert s["slope"] < 0.8


def test_calibration_slope_single_class_returns_nan():
    y = np.zeros(100, dtype=int)
    p = np.linspace(0.1, 0.9, 100)
    s = calibration_slope_intercept(y, p)
    assert np.isnan(s["slope"])
    assert "note" in s


def test_net_benefit_shapes():
    y, p = perfectly_calibrated(500)
    nb = net_benefit(y, p)
    assert len(nb["thresholds"]) == len(nb["model"]) == len(nb["treat_all"])
    assert all(v == 0.0 for v in nb["treat_none"])


def test_net_benefit_treat_all_formula():
    """全部干预的净收益应为 prev - (1-prev)*w。"""
    y, p = perfectly_calibrated(1000)
    nb = net_benefit(y, p, thresholds=np.array([0.3]))
    prev = y.mean()
    w = 0.3 / 0.7
    assert np.isclose(nb["treat_all"][0], prev - (1 - prev) * w)


def test_net_benefit_model_beats_treat_all_for_good_model():
    """好模型在合理阈值区间应优于全部干预。"""
    y, p = perfectly_calibrated(20000)
    nb = net_benefit(y, p, thresholds=np.array([0.4]))
    assert nb["model"][0] > nb["treat_all"][0]


def test_calibration_summary_keys():
    y, p = perfectly_calibrated(1000)
    s = calibration_summary(y, p)
    for k in ["ici", "e50", "e90", "mean_gap", "brier", "slope", "intercept", "calibration_curve"]:
        assert k in s


def test_e50_le_e90():
    y, p = systematically_overestimated()
    s = calibration_summary(y, p)
    assert s["e50"] <= s["e90"]


def test_perfect_calibration_summary():
    y, p = perfectly_calibrated()
    s = calibration_summary(y, p)
    assert s["ici"] < 0.02
    assert abs(s["mean_gap"]) < 0.02
