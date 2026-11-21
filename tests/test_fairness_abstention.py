"""公平性与拒识指标测试。"""

from __future__ import annotations

import numpy as np
import pytest

from clinaudit.metrics.abstention import (
    abstention_summary,
    aurc,
    confidence_score,
    conformal_prediction_sets,
    empirical_coverage,
    risk_at_coverage,
    selective_curve,
    split_conformal_quantile,
)
from clinaudit.metrics.fairness import fairness_report, make_groups, subgroup_metrics


# ------------------------------------------------------------ 公平性
def make_grouped(n_per: int = 200, auc_a: float = 0.95, auc_b: float = 0.6, seed: int = 0):
    """构造两组，判别力不同的数据。

    生成方式：对每个样本，以概率 ``strength`` 给出"正确的极端概率"，
    否则给出纯随机概率。``strength`` 越大，AUC 越高。
    """
    rng = np.random.default_rng(seed)
    ys, ps, gs = [], [], []
    for g, strength in (("A", auc_a), ("B", auc_b)):
        yv = rng.integers(0, 2, n_per)
        inform = rng.random(n_per) < strength
        # 有信息时：阳性给高概率，阴性给低概率
        informative_p = np.where(yv == 1, 0.75, 0.25)
        pv = np.where(inform, informative_p, rng.uniform(0.05, 0.95, n_per))
        pv = np.clip(pv, 0.01, 0.99)
        ys.append(yv)
        ps.append(pv)
        gs.extend([g] * n_per)
    return np.concatenate(ys), np.concatenate(ps), np.array(gs, dtype=object)


def test_subgroup_metrics_has_both_groups():
    y, p, g = make_grouped()
    m = subgroup_metrics(y, p, g)
    assert set(m) == {"A", "B"}
    assert m["A"]["n"] == 200


def test_subgroup_metrics_detects_auc_difference():
    y, p, g = make_grouped()
    m = subgroup_metrics(y, p, g)
    assert m["A"]["auc"] > m["B"]["auc"]


def test_subgroup_metrics_skips_tiny_group():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    g = np.array(["A", "A", "A", "B"], dtype=object)
    m = subgroup_metrics(y, p, g)
    assert "note" in m["B"]


def test_fairness_report_flags_large_gap():
    y, p, g = make_grouped()
    r = fairness_report(y, p, g)
    assert r["n_groups"] == 2
    assert r["diffs"]["auc_range"] > 0.1
    assert any(f["metric"] == "AUC 组间差异" for f in r["flags"])


def test_fairness_report_no_flag_when_similar():
    rng = np.random.default_rng(5)
    n = 600
    y = rng.integers(0, 2, n)
    p = np.clip(y * 0.5 + rng.random(n) * 0.5, 0.01, 0.99)
    g = np.array(["A"] * (n // 2) + ["B"] * (n - n // 2), dtype=object)
    r = fairness_report(y, p, g)
    assert r["diffs"]["auc_range"] < 0.15


def test_fairness_report_single_group():
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    g = np.array(["A"] * 5, dtype=object)
    r = fairness_report(y, p, g)
    assert "note" in r
    assert r["flags"] == []


def test_disparate_impact_ratio():
    """两组的干预率差异应体现在 DI ratio。"""
    y = np.array([0, 1] * 100)
    p = np.concatenate([np.full(100, 0.9), np.full(100, 0.1)])
    g = np.array(["A"] * 100 + ["B"] * 100, dtype=object)
    r = fairness_report(y, p, g)
    assert r["diffs"]["disparate_impact_ratio"] < 0.5


def test_make_groups_discrete():
    v = np.array([0, 1, 0, 1, 0])
    g, names = make_groups(v)
    assert set(names) == {"0", "1"}


def test_make_groups_continuous():
    v = np.arange(100, dtype=float)
    g, names = make_groups(v, n_bins=4)
    assert len(names) == 4
    assert all(str(n).startswith("Q") for n in names)


def test_make_groups_handles_nan():
    v = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0])
    g, _ = make_groups(v)
    assert len(g) == len(v)


# ------------------------------------------------------------ 拒识
def test_confidence_score_range():
    p = np.array([0.0, 0.5, 1.0])
    c = confidence_score(p)
    assert np.allclose(c, [1.0, 0.5, 1.0])


def test_selective_curve_monotone_coverage():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = rng.random(300)
    c = selective_curve(y, p)
    assert np.all(np.diff(c["coverage"]) > 0)
    assert c["coverage"][-1] == pytest.approx(1.0)


def test_selective_risk_increases_for_informative_confidence():
    """若置信度有效，全纳入时的风险应高于只留高置信样本。"""
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 2, n)
    # 高置信样本更可能预测正确
    correct_conf = np.where(y == 1, 0.9, 0.1)
    noisy = rng.random(n)
    p = np.where(noisy < 0.7, correct_conf, rng.random(n))
    c = selective_curve(y, p)
    assert c["risk"][0] < c["risk"][-1]


def test_aurc_better_than_random():
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 2, n)
    good = np.where(y == 1, 0.9, 0.1) + rng.normal(0, 0.05, n)
    good = np.clip(good, 0.01, 0.99)
    bad = rng.random(n)
    assert aurc(y, good) < aurc(y, bad)


def test_risk_at_coverage_bounds():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    r = risk_at_coverage(y, p, 0.5)
    assert 0.4 <= r["achieved_coverage"] <= 0.6
    assert 0 <= r["risk"] <= 1


def test_risk_at_coverage_invalid():
    with pytest.raises(ValueError, match="target_coverage"):
        risk_at_coverage(np.array([0, 1]), np.array([0.1, 0.9]), 1.5)


def test_conformal_coverage_meets_guarantee():
    """Split conformal 的经验覆盖率应 >= 1-alpha（允许小波动）。"""
    rng = np.random.default_rng(0)
    n_cal, n_test = 1000, 4000
    p_cal = rng.uniform(0.1, 0.9, n_cal)
    y_cal = (rng.random(n_cal) < p_cal).astype(int)
    p_test = rng.uniform(0.1, 0.9, n_test)
    y_test = (rng.random(n_test) < p_test).astype(int)

    for alpha in (0.05, 0.1, 0.2):
        sets = conformal_prediction_sets(y_cal, p_cal, p_test, alpha=alpha)
        cov = empirical_coverage(y_test, sets)
        assert cov >= 1 - alpha - 0.03, f"alpha={alpha} 覆盖率 {cov} 不足"


def test_conformal_quantile_monotone():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.1, 0.9, 500)
    y = (rng.random(500) < p).astype(int)
    q1 = split_conformal_quantile(y, p, alpha=0.05)
    q2 = split_conformal_quantile(y, p, alpha=0.2)
    assert q1 >= q2


def test_conformal_quantile_too_small():
    with pytest.raises(ValueError, match="过小"):
        split_conformal_quantile(np.array([1]), np.array([0.5]))


def test_conformal_set_size():
    rng = np.random.default_rng(0)
    p_cal = rng.uniform(0.1, 0.9, 300)
    y_cal = (rng.random(300) < p_cal).astype(int)
    p_test = rng.uniform(0.1, 0.9, 100)
    s = conformal_prediction_sets(y_cal, p_cal, p_test, alpha=0.1)
    assert len(s["set_size"]) == 100
    assert 1.0 <= s["mean_set_size"] <= 2.0


def test_abstention_summary_keys():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = rng.random(400)
    s = abstention_summary(y, p)
    for k in ["base_risk", "aurc", "curve", "at_coverage"]:
        assert k in s
    assert len(s["at_coverage"]) == 5
