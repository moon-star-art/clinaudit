"""CLI 各子命令的端到端测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from clinaudit.cli import _cache, _conclusion, _figures_for, _sections, build_parser, main
from clinaudit.data.loaders import load_heart
from clinaudit.models.training import train_model
from clinaudit.audit import audit


@pytest.fixture(scope="module")
def demo_cache():
    """提供一个可用的缓存目录（用内置数据即可，无需联网）。"""
    return Path(__file__).resolve().parents[1] / "_build" / "testtmp_session" / "cli"


def test_cache_helper(demo_cache):
    class A:
        cache = str(demo_cache)

    assert _cache(A()) == demo_cache


def test_cache_helper_default():
    class A:
        cache = None

    assert _cache(A()) == Path.cwd() / "data_cache"


def test_parser_all_subcommands():
    p = build_parser()
    for argv, fn in [
        (["demo"], "cmd_demo"),
        (["audit"], "cmd_audit"),
        (["external"], "cmd_external"),
        (["calibrate"], "cmd_calibrate"),
        (["datasets"], "cmd_datasets"),
    ]:
        assert p.parse_args(argv).func.__name__ == fn


def test_parser_top_level_cache():
    a = build_parser().parse_args(["--cache", "X", "demo"])
    assert a.cache == "X"


def test_parser_allow_network_flag():
    a = build_parser().parse_args(["--allow-network", "demo"])
    assert a.allow_network is True


def test_cli_datasets_output(capsys):
    assert main(["datasets"]) == 0
    out = capsys.readouterr().out
    assert "cleveland" in out
    assert "wdbc" in out


def test_sections_structure(demo_cache, tmp_dir):
    ds = load_heart(cache_dir=demo_cache, allow_network=False)
    m = train_model(ds.X, ds.y, model="lr", seed=0)
    p = m.predict_proba(ds.X)
    res = audit(ds.y, p, X=ds.X, centers=ds.center)
    secs = _sections(res)
    titles = [s["title"] for s in secs]
    assert "判别力" in titles
    assert "校准" in titles
    assert "临床效用（净收益）" in titles
    assert "拒识能力" in titles
    for s in secs:
        assert "columns" in s["table"]
        assert "rows" in s["table"]


def test_sections_include_subgroups(demo_cache):
    ds = load_heart(cache_dir=demo_cache, allow_network=False)
    m = train_model(ds.X, ds.y, model="lr", seed=0)
    p = m.predict_proba(ds.X)
    res = audit(ds.y, p, X=ds.X, centers=ds.center)
    assert any(s["title"] == "亚组表现" for s in _sections(res))


def test_figures_all_rendered(demo_cache):
    ds = load_heart(cache_dir=demo_cache, allow_network=False)
    m = train_model(ds.X, ds.y, model="lr", seed=0)
    p = m.predict_proba(ds.X)
    res = audit(ds.y, p, X=ds.X, centers=ds.center)
    figs = _figures_for(res)
    assert "校准曲线" in figs
    assert all(v.startswith("iVBOR") for v in figs.values())


def test_conclusion_contains_key_numbers(demo_cache):
    ds = load_heart(cache_dir=demo_cache, allow_network=False)
    m = train_model(ds.X, ds.y, model="lr", seed=0)
    p = m.predict_proba(ds.X)
    res = audit(ds.y, p, X=ds.X, centers=ds.center)
    c = _conclusion(res)
    assert f"{res.score:.1f}" in c
    assert res.grade in c


def test_conclusion_without_flags():
    from clinaudit.audit import AuditResult

    res = AuditResult(
        dataset_name="toy",
        model_name="lr",
        n=100,
        prevalence=0.3,
        discrimination={"auc": 0.8},
        calibration={"ici": 0.02},
        fairness_center=None,
        fairness_sex=None,
        decision_curve={"thresholds": [], "model": [], "treat_all": [], "treat_none": []},
        abstention={"base_risk": 0.2, "aurc": 0.1, "at_coverage": []},
        score=85.0,
        grade="A（可信）",
        flags=[],
    )
    c = _conclusion(res)
    assert "未发现显著风险提示" in c


def test_cli_external_writes_json(demo_cache, tmp_dir, capsys):
    js = tmp_dir / "ext.json"
    rc = main(["--cache", str(demo_cache), "external", "--train", "cleveland", "--json", str(js)])
    assert rc == 0
    data = json.loads(js.read_text(encoding="utf-8"))
    assert "centers" in data
    assert len(data["centers"]) == 4


def test_cli_external_bad_center(demo_cache, capsys):
    rc = main(["--cache", str(demo_cache), "external", "--train", "mars"])
    assert rc == 1


def test_cli_audit(demo_cache, tmp_dir, capsys):
    out = tmp_dir / "audit.html"
    js = tmp_dir / "audit.json"
    rc = main(
        ["--cache", str(demo_cache), "audit", "--data", "heart", "--cv", "3",
         "--out", str(out), "--json", str(js)]
    )
    assert rc == 0
    assert out.exists()
    assert js.exists()
    d = json.loads(js.read_text(encoding="utf-8"))
    assert "discrimination" in d


def test_cli_calibrate(demo_cache, tmp_dir, capsys):
    js = tmp_dir / "cal.json"
    rc = main(
        ["--cache", str(demo_cache), "calibrate", "--data", "heart",
         "--method", "sigmoid", "--json", str(js)]
    )
    assert rc == 0
    d = json.loads(js.read_text(encoding="utf-8"))
    assert "raw" in d and "sigmoid" in d
    assert "ici" in d["raw"]


def test_cli_calibrate_isotonic(demo_cache, tmp_dir, capsys):
    rc = main(["--cache", str(demo_cache), "calibrate", "--data", "heart", "--method", "isotonic"])
    assert rc == 0


def test_cli_demo_writes_report(demo_cache, tmp_dir, capsys):
    out = tmp_dir / "demo.html"
    rc = main(["--cache", str(demo_cache), "demo", "--out", str(out)])
    assert rc == 0
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "临床预测模型可信度审计报告" in html
    assert "data:image/png;base64," in html


def test_cli_main_returns_int():
    assert isinstance(main(["datasets"]), int)
