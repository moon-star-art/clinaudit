"""数据加载测试：解析正确性、缺失值处理、多中心拆分、离线兜底。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clinaudit.data.loaders import (
    HEART_CENTERS,
    HEART_COLUMNS,
    ClinicalDataset,
    _find_data_dir,
    _read_heart_table,
    load_heart,
    merge_centers,
    parse_heart,
)


CLEVELAND_SAMPLE = """63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0
67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2
67.0,1.0,4.0,120.0,229.0,0.0,2.0,129.0,1.0,2.6,2.0,2.0,7.0,1
"""

HUNGARIAN_SAMPLE = """28,1,2,130,132,0,2,185,0,0,?,?,?,0
29,1,2,120,243,0,0,160,0,0,?,?,?,0
"""


@pytest.fixture
def heart_dir(tmp_dir):
    d = tmp_dir / "heart"
    d.mkdir()
    (d / "processed.cleveland.data").write_text(CLEVELAND_SAMPLE, encoding="latin-1")
    (d / "processed.hungarian.data").write_text(HUNGARIAN_SAMPLE, encoding="latin-1")
    return d


def test_read_heart_table_shape(heart_dir):
    df = _read_heart_table(heart_dir / "processed.cleveland.data")
    assert df.shape == (3, 14)
    assert list(df.columns) == HEART_COLUMNS


def test_read_heart_table_question_marks_become_nan(heart_dir):
    df = _read_heart_table(heart_dir / "processed.hungarian.data")
    # 后 3 列在样本里是 ?
    assert df["ca"].isna().all()
    assert df["thal"].isna().all()


def test_read_heart_table_numeric_dtypes(heart_dir):
    df = _read_heart_table(heart_dir / "processed.cleveland.data")
    assert df["age"].dtype.kind == "f"
    assert df["age"].iloc[0] == 63.0


def test_parse_heart_labels_binarized(heart_dir):
    parts = parse_heart(heart_dir)
    assert set(parts) == {"cleveland", "hungarian"}
    # num=0 -> 0；num=2 -> 1；num=1 -> 1
    assert parts["cleveland"].y.tolist() == [0, 1, 1]


def test_parse_heart_drops_num_column(heart_dir):
    parts = parse_heart(heart_dir)
    assert "num" not in parts["cleveland"].X.columns
    assert parts["cleveland"].n_features == 13


def test_parse_heart_center_column(heart_dir):
    parts = parse_heart(heart_dir)
    assert set(parts["cleveland"].center.tolist()) == {"cleveland"}
    assert set(parts["hungarian"].center.tolist()) == {"hungarian"}


def test_parse_heart_skips_missing_files(heart_dir):
    (heart_dir / "processed.hungarian.data").unlink()
    parts = parse_heart(heart_dir)
    assert set(parts) == {"cleveland"}


def test_merge_centers_concatenates(heart_dir):
    parts = parse_heart(heart_dir)
    merged = merge_centers(parts)
    assert len(merged) == 5
    assert merged.center is not None
    assert set(merged.center.tolist()) == {"cleveland", "hungarian"}


def test_merge_centers_empty_raises():
    with pytest.raises(ValueError, match="没有可合并"):
        merge_centers({})


def test_by_center_splits(heart_dir):
    parts = parse_heart(heart_dir)
    merged = merge_centers(parts)
    split = merged.by_center()
    assert set(split) == {"cleveland", "hungarian"}
    assert len(split["cleveland"]) == 3
    assert len(split["hungarian"]) == 2


def test_subset_preserves_center(heart_dir):
    parts = parse_heart(heart_dir)
    merged = merge_centers(parts)
    sub = merged.subset(np.array([True, False, False, False, False]))
    assert len(sub) == 1
    assert sub.center is not None


def test_summary_keys(heart_dir):
    parts = parse_heart(heart_dir)
    s = parts["cleveland"].summary()
    for k in ["name", "n_samples", "n_features", "prevalence", "missing_fraction"]:
        assert k in s


def test_prevalence_calculation(heart_dir):
    parts = parse_heart(heart_dir)
    assert np.isclose(parts["cleveland"].prevalence, 2 / 3)


def test_load_heart_offline_uses_bundled(tmp_dir):
    """无缓存 + 无网络时，应退回随包内置数据。"""
    ds = load_heart(cache_dir=tmp_dir, allow_network=False, use_bundled=True)
    assert len(ds) > 500
    assert ds.center is not None
    assert set(ds.center.tolist()) == set(HEART_CENTERS)
    assert ds.n_features == 13


def test_load_heart_bundled_single_center(tmp_dir):
    ds = load_heart(cache_dir=tmp_dir, allow_network=False, center="cleveland")
    assert set(ds.center.tolist()) == {"cleveland"}
    assert len(ds) == 303


def test_load_heart_unknown_center(tmp_dir):
    with pytest.raises(KeyError, match="未知中心"):
        load_heart(cache_dir=tmp_dir, allow_network=False, center="mars")


def test_load_heart_no_bundled_raises(tmp_dir):
    with pytest.raises(FileNotFoundError):
        load_heart(cache_dir=tmp_dir, allow_network=False, use_bundled=False)


def test_bundled_centers_have_expected_sizes(tmp_dir):
    """内置数据的四个中心样本数应与 UCI 文档一致。"""
    ds = load_heart(cache_dir=tmp_dir, allow_network=False)
    parts = ds.by_center()
    for k, (_full, expected) in HEART_CENTERS.items():
        assert len(parts[k]) == expected, f"{k} 样本数不符"


def test_find_data_dir_flat(tmp_dir):
    d = tmp_dir / "flat"
    d.mkdir()
    (d / "processed.cleveland.data").write_text("x", encoding="utf-8")
    assert _find_data_dir(d, "heart") == d


def test_find_data_dir_nested(tmp_dir):
    d = tmp_dir / "nested"
    inner = d / "sub"
    inner.mkdir(parents=True)
    (inner / "processed.cleveland.data").write_text("x", encoding="utf-8")
    assert _find_data_dir(d, "heart") == inner


def test_find_data_dir_fallback(tmp_dir):
    d = tmp_dir / "empty"
    d.mkdir()
    assert _find_data_dir(d, "heart") == d


def test_clinical_dataset_basic():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    ds = ClinicalDataset(name="t", X=X, y=np.array([0, 1]))
    assert len(ds) == 2
    assert ds.n_features == 2
    assert ds.prevalence == 0.5
    assert ds.by_center() == {"t": ds}


def test_heart_has_real_missingness(tmp_dir):
    """Hungarian/Switzerland/VA 有大量缺失，这一点必须在数据层保留。"""
    ds = load_heart(cache_dir=tmp_dir, allow_network=False)
    parts = ds.by_center()
    assert parts["cleveland"].X.isna().mean().mean() < 0.01
    assert parts["va"].X.isna().mean().mean() > 0.15
