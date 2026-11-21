"""模型训练、审计流程、报告与 CLI 的测试。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from clinaudit.audit import _trust_score, audit, cross_center_audit
from clinaudit.cli import build_parser, main
from clinaudit.data.loaders import load_heart
from clinaudit.models.training import (
    TrainedModel,
    build_pipeline,
    cross_val_predict_proba,
    train_model,
    train_with_calibration,
)
from clinaudit.report import (
    plot_calibration,
    plot_decision_curve,
    plot_roc,
    plot_selective,
    plot_subgroup_bars,
    render_report,
)


def make_synthetic_data(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "age": rng.integers(30, 80, n).astype(float),
            "sex": rng.integers(0, 2, n).astype(float),
            "chol": rng.normal(240, 40, n),
        }
    )
    logit = -4 + 0.05 * X["age"] + 0.01 * X["chol"] + 0.6 * X["sex"]
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    return X, y


# ------------------------------------------------------------ 训练
def test_build_pipeline_unknown_model():
    with pytest.raises(ValueError, match="未知模型"):
        build_pipeline(model="transformer")


@pytest.mark.parametrize("model", ["lr", "gbdt", "rf"])
def test_build_pipeline_all_models(model):
    X, y = make_synthetic_data(300)
    pipe = build_pipeline(model=model, seed=0)
    pipe.fit(X, y)
    p = pipe.predict_proba(X)[:, 1]
    assert p.shape == (300,)
    assert np.all((p >= 0) & (p <= 1))


def test_train_model_predicts():
    X, y = make_synthetic_data()
    m = train_model(X, y, model="lr", seed=0)
    p = m.predict_proba(X)
    assert p.shape == (len(y),)
    assert m.model_name == "lr"


def test_train_model_handles_missing_values():
    """含缺失值时流水线应正常工作（中位数填补 + 指示列）。"""
    X, y = make_synthetic_data(400)
    X = X.copy()
    X.loc[X.index[:100], "chol"] = np.nan
    m = train_model(X, y, model="lr", seed=0)
    assert np.all(np.isfinite(m.predict_proba(X)))


def test_cross_val_predict_returns_all():
    X, y = make_synthetic_data(500)
    p = cross_val_predict_proba(X, y, model="lr", n_splits=5, seed=0)
    assert p.shape == (500,)
    assert np.all((p > 0) & (p < 1))


def test_cross_val_reproducible():
    X, y = make_synthetic_data(400)
    a = cross_val_predict_proba(X, y, n_splits=4, seed=0)
    b = cross_val_predict_proba(X, y, n_splits=4, seed=0)
    assert np.allclose(a, b)


def test_cross_val_too_few_splits():
    """某一类只有 1 个样本时无法分层切分，应抛出明确错误。"""
    X, y = make_synthetic_data(20)
    y = np.zeros(20, dtype=int)
    y[0] = 1  # 只有 1 个正例
    with pytest.raises(ValueError, match="交叉验证"):
        cross_val_predict_proba(X, y, n_splits=5)


def test_cross_val_reduces_splits_when_minority_small():
    """少数类只有 2 例时，折数应自动降为 2 而非报错。"""
    X, y = make_synthetic_data(40)
    y = np.zeros(40, dtype=int)
    y[:2] = 1
    p = cross_val_predict_proba(X, y, n_splits=5, seed=0)
    assert p.shape == (40,)


def test_train_with_calibration_both_methods():
    X, y = make_synthetic_data(600)
    for method in ("sigmoid", "isotonic"):
        m = train_with_calibration(X, y, model="lr", method=method, seed=0)
        assert method in m.model_name
        p = m.predict_proba(X)
        assert np.all((p >= 0) & (p <= 1))


def test_train_with_calibration_invalid_method():
    X, y = make_synthetic_data(200)
    with pytest.raises(ValueError, match="sigmoid"):
        train_with_calibration(X, y, method="magic")


# ------------------------------------------------------------ 审计
def test_audit_basic():
    X, y = make_synthetic_data(600)
    m = train_model(X, y, model="lr", seed=0)
    p = m.predict_proba(X)
    res = audit(y, p, dataset_name="toy", model_name="lr", X=X)
    assert res.n == 600
    assert 0 <= res.discrimination["auc"] <= 1
    assert np.isfinite(res.calibration["ici"])
    assert np.isfinite(res.score)
    assert res.grade


def test_audit_with_centers_produces_subgroups():
    X, y = make_synthetic_data(600)
    centers = np.array(["A"] * 300 + ["B"] * 300, dtype=object)
    m = train_model(X, y, model="lr", seed=0)
    p = m.predict_proba(X)
    res = audit(y, p, X=X, centers=centers)
    assert res.fairness_center is not None
    assert len(res.fairness_center["groups"]) == 2
    assert len(res.subgroup_table) >= 2


def test_audit_with_sex():
    X, y = make_synthetic_data(600)
    m = train_model(X, y, model="lr", seed=0)
    p = m.predict_proba(X)
    res = audit(y, p, X=X)
    assert res.fairness_sex is not None


def test_audit_to_dict_serializable():
    X, y = make_synthetic_data(400)
    m = train_model(X, y, model="lr", seed=0)
    p = m.predict_proba(X)
    res = audit(y, p, X=X)
    d = res.to_dict()
    # 必须可 JSON 序列化
    s = json.dumps(d, ensure_ascii=False)
    assert len(s) > 100
    assert "discrimination" in d


def test_audit_flags_on_bad_calibration():
    """严重高估的模型应触发校准红旗。"""
    rng = np.random.default_rng(0)
    n = 1000
    y = rng.integers(0, 2, n)
    p = np.clip(np.where(y == 1, 0.6, 0.55) + rng.normal(0, 0.02, n), 0.01, 0.99)
    res = audit(y, p, dataset_name="bad")
    assert any("校准" in f["metric"] for f in res.flags)


def test_trust_score_range():
    disc = {"auc": 0.9}
    cal = {"ici": 0.02}
    ab = {"aurc": 0.05, "base_risk": 0.1}
    score, grade = _trust_score(disc, cal, None, ab)
    assert 0 <= score <= 100
    assert grade.startswith(("A", "B", "C", "D"))


def test_trust_score_empty():
    score, grade = _trust_score({}, {}, None, {})
    assert np.isnan(score)


def test_trust_score_good_beats_bad():
    good, _ = _trust_score({"auc": 0.95}, {"ici": 0.01}, {"diffs": {"auc_range": 0.02}},
                           {"aurc": 0.03, "base_risk": 0.1})
    bad, _ = _trust_score({"auc": 0.6}, {"ici": 0.3}, {"diffs": {"auc_range": 0.3}},
                          {"aurc": 0.28, "base_risk": 0.3})
    assert good > bad


def test_cross_center_audit(tmp_dir):
    ds = load_heart(cache_dir=tmp_dir, allow_network=False)
    parts = ds.by_center()
    m = train_model(parts["cleveland"].X, parts["cleveland"].y, model="lr", seed=0)
    ext = cross_center_audit(m, parts)
    assert len(ext["centers"]) == 4
    for row in ext["centers"]:
        assert "auc" in row and "ici" in row


def test_cross_center_reveals_drift(tmp_dir):
    """Cleveland 与其他中心的表现应有明显差异（这是本工具的核心论点）。"""
    ds = load_heart(cache_dir=tmp_dir, allow_network=False)
    parts = ds.by_center()
    m = train_model(parts["cleveland"].X, parts["cleveland"].y, model="lr", seed=0)
    ext = cross_center_audit(m, parts)
    by = {r["center"]: r for r in ext["centers"]}
    assert by["cleveland"]["auc"] > by["switzerland"]["auc"]


# ------------------------------------------------------------ 报告
def test_plot_functions_return_png():
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    from clinaudit.metrics.calibration import calibration_curve_data, net_benefit
    from clinaudit.metrics.discrimination import roc_curve_points
    from clinaudit.metrics.abstention import selective_curve

    assert plot_calibration(calibration_curve_data(y, p)["bins"]).startswith("iVBOR")
    assert plot_roc(roc_curve_points(y, p), auc=0.8).startswith("iVBOR")
    assert plot_decision_curve(net_benefit(y, p)).startswith("iVBOR")
    assert plot_selective(selective_curve(y, p)).startswith("iVBOR")


def test_plot_subgroup_bars():
    groups = {"A": {"auc": 0.9}, "B": {"auc": 0.7}}
    assert plot_subgroup_bars(groups).startswith("iVBOR")


def test_render_report(tmp_dir):
    out = tmp_dir / "r.html"
    path = render_report(
        out,
        title="测试报告",
        overview={"样本数": 100, "AUC": 0.85},
        sections=[{"title": "第一节", "table": {"columns": ["a", "b"], "rows": [[1, 2]]}, "note": "备注"}],
        flags=[{"level": "高", "metric": "AUC 差异", "value": 0.3, "message": "测试红旗"}],
        conclusion="结论文字",
    )
    html = path.read_text(encoding="utf-8")
    assert "测试报告" in html
    assert "测试红旗" in html
    assert "结论文字" in html


def test_render_report_creates_dirs(tmp_dir):
    out = tmp_dir / "a" / "b" / "r.html"
    render_report(out, title="T", overview={}, sections=[])
    assert out.exists()


# ------------------------------------------------------------ CLI 解析
def test_parser_requires_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_parser_demo():
    a = build_parser().parse_args(["demo"])
    assert a.func.__name__ == "cmd_demo"


def test_parser_audit():
    a = build_parser().parse_args(["audit", "--data", "pima", "--model", "gbdt"])
    assert a.data == "pima" and a.model == "gbdt"


def test_parser_external():
    a = build_parser().parse_args(["external", "--train", "va"])
    assert a.train == "va"


def test_parser_calibrate():
    a = build_parser().parse_args(["calibrate", "--method", "isotonic"])
    assert a.method == "isotonic"


def test_cli_datasets(capsys):
    assert main(["datasets"]) == 0
    out = capsys.readouterr().out
    assert "heart" in out and "pima" in out


def test_cli_external(tmp_dir, capsys):
    # 顶层参数 --cache 必须放在子命令之前
    rc = main(["--cache", str(tmp_dir), "external", "--train", "cleveland"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "cleveland" in out and "AUC" in out
