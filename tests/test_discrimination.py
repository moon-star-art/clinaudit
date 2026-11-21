"""判别力指标测试。"""

from __future__ import annotations

import numpy as np
import pytest

from clinaudit.metrics.discrimination import (
    best_youden_threshold,
    confusion_at_threshold,
    discrimination_summary,
    pr_auc,
    roc_auc,
    roc_curve_points,
)


def test_roc_auc_perfect():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_auc(y, p) == 1.0


def test_roc_auc_worst():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    assert roc_auc(y, p) == 0.0


def test_roc_auc_constant_is_half():
    y = np.array([0, 1, 0, 1])
    p = np.full(4, 0.5)
    assert np.isclose(roc_auc(y, p), 0.5)


def test_roc_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.4 + rng.random(500) * 0.6, 0, 1)
    assert np.isclose(roc_auc(y, p), roc_auc_score(y, p))


def test_roc_auc_handles_ties_correctly():
    from sklearn.metrics import roc_auc_score

    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.5, 0.5, 0.5, 0.5, 0.8, 0.2])
    assert np.isclose(roc_auc(y, p), roc_auc_score(y, p))


def test_roc_auc_single_class():
    y = np.zeros(10, dtype=int)
    p = np.linspace(0, 1, 10)
    assert np.isnan(roc_auc(y, p))


def test_roc_auc_length_mismatch():
    with pytest.raises(ValueError, match="长度不一致"):
        roc_auc(np.array([0, 1]), np.array([0.1]))


def test_roc_curve_monotone():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    c = roc_curve_points(y, p)
    assert np.all(np.diff(c["fpr"]) >= -1e-12)
    assert np.all(np.diff(c["tpr"]) >= -1e-12)
    assert c["fpr"][0] == 0.0 and c["tpr"][0] == 0.0
    assert np.isclose(c["fpr"][-1], 1.0)
    assert np.isclose(c["tpr"][-1], 1.0)


def test_pr_auc_perfect():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert np.isclose(pr_auc(y, p), 1.0)


def test_pr_auc_matches_sklearn():
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.3 + rng.random(400) * 0.7, 0, 1)
    assert np.isclose(pr_auc(y, p), average_precision_score(y, p), atol=1e-6)


def test_pr_auc_no_positive():
    y = np.zeros(10, dtype=int)
    assert np.isnan(pr_auc(y, np.random.default_rng(0).random(10)))


def test_confusion_matrix_counts():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1])
    m = confusion_at_threshold(y, p, 0.5)
    assert m["tp"] == 1
    assert m["fn"] == 1
    assert m["fp"] == 1
    assert m["tn"] == 1


def test_confusion_derived_metrics():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.6, 0.4, 0.1])
    m = confusion_at_threshold(y, p, 0.5)
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["youden_j"] == 1.0


def test_youden_threshold_finds_boundary():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    best = best_youden_threshold(y, p)
    assert np.isclose(best["youden_j"], 1.0)
    assert 0.4 < best["threshold"] <= 0.6


def test_discrimination_summary_keys():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    s = discrimination_summary(y, p)
    for k in ["auc", "pr_auc", "sensitivity", "specificity", "roc_curve", "n_positive"]:
        assert k in s


def test_discrimination_summary_prevalence():
    y = np.array([1, 1, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2])
    s = discrimination_summary(y, p)
    assert np.isclose(s["prevalence"], 0.75)
