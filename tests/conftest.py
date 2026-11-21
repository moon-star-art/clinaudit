"""共享测试夹具（同 motifscope：避免 tmp_path 在受限沙箱下的清理失败）。"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

_TMP_ROOT = Path(__file__).resolve().parents[1] / "_build" / "testtmp"


@pytest.fixture
def tmp_dir() -> Path:
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    d = _TMP_ROOT / uuid.uuid4().hex[:12]
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def tmp_path_factory():
    class _Factory:
        _root = Path(__file__).resolve().parents[1] / "_build" / "testtmp_session"

        def mktemp(self, basename: str | None = None, numbered: bool = True) -> Path:
            self._root.mkdir(parents=True, exist_ok=True)
            d = self._root / (basename or f"tmp{uuid.uuid4().hex[:8]}")
            d.mkdir(parents=True, exist_ok=True)
            return d

    return _Factory()
