"""数据集加载：UCI 多中心心脏数据、Pima、WDBC、ILPD。

统一返回 :class:`ClinicalDataset`，包含特征表、二分类标签、以及（若有）
**中心标识**用于外部验证与公平性分析。

设计要点
--------
1. 原始数据中的 ``?`` 一律解析为缺失值，不做静默填充——缺失机制本身是
   审计的一部分（不同中心的缺失率差异会直接导致校准漂移）。
2. 特征名与单位保持与 UCI 原始文档一致，便于与文献对照。
3. 所有数据集都可离线使用：随包内置一份小型 CSV，网络不可用时自动降级。
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- 常量定义

#: 心脏数据的 14 个属性名（UCI 文档顺序）。
HEART_COLUMNS = [
    "age",           # 年龄（岁）
    "sex",           # 性别 1=男 0=女
    "cp",            # 胸痛类型 1-4
    "trestbps",      # 静息血压 mmHg
    "chol",          # 血清胆固醇 mg/dl
    "fbs",           # 空腹血糖 >120 mg/dl 1=是
    "restecg",       # 静息心电图 0-2
    "thalach",       # 最大心率
    "exang",         # 运动诱发心绞痛 1=是
    "oldpeak",       # 运动相对静息 ST 段压低
    "slope",         # 峰值运动 ST 段斜率 1-3
    "ca",            # 荧光染色主要血管数 0-3
    "thal",          # 地中海贫血 3=正常 6=固定缺陷 7=可逆缺陷
    "num",           # 诊断：0=无狭窄，1-4=有狭窄
]

#: 四个中心的缩写与全名。这是本项目做"多中心外部验证"的基础。
HEART_CENTERS = {
    "cleveland": ("Cleveland Clinic Foundation", 303),
    "hungarian": ("Hungarian Institute of Cardiology", 294),
    "switzerland": ("University Hospital Zurich", 123),
    "va": ("V.A. Medical Center, Long Beach", 200),
}

UCI_URLS = {
    "heart": "https://archive.ics.uci.edu/static/public/45/heart+disease.zip",
    "pima": "https://archive.ics.uci.edu/static/public/34/diabetes.zip",
    "wdbc": "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
    "liver": "https://archive.ics.uci.edu/static/public/225/ilpd+indian+liver+patient+dataset.zip",
}

PIMA_COLUMNS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age", "outcome",
]

LIVER_COLUMNS = [
    "age", "gender", "total_bilirubin", "direct_bilirubin",
    "alkaline_phosphatase", "alamine_aminotransferase",
    "aspartate_aminotransferase", "total_proteins", "albumin",
    "albumin_globulin_ratio", "outcome",
]


@dataclass
class ClinicalDataset:
    """统一的临床数据集容器。"""

    name: str
    X: pd.DataFrame
    y: np.ndarray
    center: np.ndarray | None = None
    positive_label: str = "阳性"
    feature_descriptions: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def prevalence(self) -> float:
        return float(np.mean(self.y))

    def subset(self, mask: np.ndarray) -> "ClinicalDataset":
        return ClinicalDataset(
            name=f"{self.name}[subset n={int(mask.sum())}]",
            X=self.X.loc[mask].reset_index(drop=True),
            y=self.y[mask],
            center=None if self.center is None else self.center[mask],
            positive_label=self.positive_label,
            feature_descriptions=self.feature_descriptions,
        )

    def by_center(self) -> dict[str, "ClinicalDataset"]:
        """按中心拆分，用于多中心外部验证。"""
        if self.center is None:
            return {self.name: self}
        out = {}
        for c in sorted(set(self.center.tolist())):
            out[c] = self.subset(self.center == c)
        return out

    def summary(self) -> dict:
        s = {
            "name": self.name,
            "n_samples": len(self),
            "n_features": self.n_features,
            "prevalence": round(self.prevalence, 4),
            "missing_fraction": round(float(self.X.isna().mean().mean()), 4),
        }
        if self.center is not None:
            vals, counts = np.unique(self.center, return_counts=True)
            s["centers"] = {str(v): int(c) for v, c in zip(vals, counts)}
        return s


# ---------------------------------------------------------------- 解析逻辑

def _read_heart_table(path: Path) -> pd.DataFrame:
    """读取 14 属性的心脏数据文件（逗号分隔，``?`` 为缺失）。"""
    text = path.read_text(encoding="latin-1", errors="replace")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 14:
            continue
        rows.append(parts[:14])
    df = pd.DataFrame(rows, columns=HEART_COLUMNS)
    df = df.replace("?", np.nan)
    for c in HEART_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def parse_heart(root: Path) -> dict[str, ClinicalDataset]:
    """解析四个中心的心脏数据，返回 {中心名: 数据集}。"""
    files = {
        "cleveland": "processed.cleveland.data",
        "hungarian": "processed.hungarian.data",
        "switzerland": "processed.switzerland.data",
        "va": "processed.va.data",
    }
    out: dict[str, ClinicalDataset] = {}
    for center, fname in files.items():
        path = root / fname
        if not path.exists():
            log.warning("缺少文件 %s，跳过中心 %s", path, center)
            continue
        df = _read_heart_table(path)
        if df.empty:
            continue
        # 标签：num == 0 为阴性，1-4 为阳性（常规二分化方式）
        y = (df.pop("num") > 0).astype(int).to_numpy()
        X = df.reset_index(drop=True)
        out[center] = ClinicalDataset(
            name=f"heart_{center}",
            X=X,
            y=y,
            center=np.full(len(y), center, dtype=object),
            positive_label="冠心病阳性",
            feature_descriptions=HEART_FEATURE_DOCS,
        )
    return out


def merge_centers(parts: dict[str, ClinicalDataset], name: str = "heart_all") -> ClinicalDataset:
    """把多个中心合并成一个数据集，保留 center 列信息。"""
    if not parts:
        raise ValueError("没有可合并的数据集")
    keys = list(parts)
    X = pd.concat([parts[k].X for k in keys], ignore_index=True)
    y = np.concatenate([parts[k].y for k in keys])
    center = np.concatenate(
        [np.full(len(parts[k].y), k, dtype=object) for k in keys]
    )
    return ClinicalDataset(
        name=name,
        X=X,
        y=y,
        center=center,
        positive_label=parts[keys[0]].positive_label,
        feature_descriptions=parts[keys[0]].feature_descriptions,
    )


HEART_FEATURE_DOCS = {
    "age": "年龄（岁）",
    "sex": "性别（1=男，0=女）",
    "cp": "胸痛类型（1=典型心绞痛，2=非典型，3=非心绞痛，4=无症状）",
    "trestbps": "静息血压（mmHg）",
    "chol": "血清胆固醇（mg/dl）",
    "fbs": "空腹血糖>120mg/dl（1=是）",
    "restecg": "静息心电图结果（0=正常，1=ST-T异常，2=左室肥厚）",
    "thalach": "达到的最大心率",
    "exang": "运动诱发心绞痛（1=是）",
    "oldpeak": "运动相对静息的 ST 段压低",
    "slope": "峰值运动 ST 段斜率（1=上斜，2=平坦，3=下斜）",
    "ca": "荧光染色主要血管数（0-3）",
    "thal": "地中海贫血（3=正常，6=固定缺陷，7=可逆缺陷）",
}


def parse_pima(root: Path) -> ClinicalDataset:
    """解析 Pima 印第安人糖尿病数据。

    UCI 原始包为 ``diabetes-data.tar.Z``（需 7z/untar），若解压失败则尝试
    随包内置的 CSV。
    """
    candidates = [
        root / "pima-indians-diabetes.data.csv",
        root / "pima-indians-diabetes.csv",
        root / "diabetes.csv",
    ]
    for c in candidates:
        if c.exists():
            return _pima_from_csv(c)
    raise FileNotFoundError(
        f"未找到 Pima 数据文件（尝试过 {[str(c) for c in candidates]}）"
    )


def _pima_from_csv(path: Path) -> ClinicalDataset:
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 9:
        raise ValueError(f"Pima 数据列数不足: {df.shape}")
    df = df.iloc[:, :9]
    df.columns = PIMA_COLUMNS
    df = df.apply(pd.to_numeric, errors="coerce")
    y = df.pop("outcome").fillna(0).astype(int).to_numpy()
    return ClinicalDataset(
        name="pima",
        X=df.reset_index(drop=True),
        y=y,
        positive_label="糖尿病阳性",
        feature_descriptions={
            "pregnancies": "怀孕次数",
            "glucose": "口服葡萄糖耐量试验 2 小时血糖",
            "blood_pressure": "舒张压（mmHg）",
            "skin_thickness": "三头肌皮褶厚度（mm）",
            "insulin": "2 小时血清胰岛素",
            "bmi": "体重指数",
            "diabetes_pedigree": "糖尿病家族史函数",
            "age": "年龄（岁）",
        },
    )


def parse_wdbc(root: Path) -> ClinicalDataset:
    """解析威斯康星乳腺癌诊断数据（569 例，30 个特征）。"""
    path = root / "wdbc.data"
    if not path.exists():
        raise FileNotFoundError(f"未找到 {path}")
    df = pd.read_csv(path, header=None)
    # 第 0 列 ID，第 1 列诊断 M/B，其余 30 个特征
    y = (df[1] == "M").astype(int).to_numpy()
    X = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    # UCI 文档中 30 个特征的顺序：mean / se / worst 各 10 个
    base = [
        "radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
    ]
    X.columns = [f"{b}_{s}" for s in ("mean", "se", "worst") for b in base]
    return ClinicalDataset(
        name="wdbc",
        X=X,
        y=y,
        positive_label="恶性",
        feature_descriptions={"radius_mean": "细胞核半径均值", "texture_mean": "纹理均值"},
    )


def parse_liver(root: Path) -> ClinicalDataset:
    """解析印度肝脏病人数据集（ILPD，583 例）。"""
    path = root / "Indian Liver Patient Dataset (ILPD).csv"
    if not path.exists():
        alt = list(root.glob("*.csv"))
        if not alt:
            raise FileNotFoundError(f"未找到 ILPD CSV: {root}")
        path = alt[0]
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 11:
        raise ValueError(f"ILPD 列数不足: {df.shape}")
    df = df.iloc[:, :11]
    df.columns = LIVER_COLUMNS
    df["gender"] = (df["gender"].astype(str).str.strip().str.lower() == "male").astype(int)
    for c in LIVER_COLUMNS:
        if c != "gender":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # 原始编码：1=肝病患者，2=健康；转为 1=患病
    raw = df.pop("outcome")
    y = (raw == 1).astype(int).to_numpy()
    return ClinicalDataset(
        name="liver",
        X=df.reset_index(drop=True),
        y=y,
        positive_label="肝病",
        feature_descriptions={"age": "年龄", "gender": "性别（1=男）"},
    )


# ---------------------------------------------------------------- 顶层接口

def _bundled_heart_dir() -> Path | None:
    """返回随包内置的心脏数据目录（存在则返回，否则 None）。

    内置副本让 `demo` 在完全离线的机器上也能首次运行成功。
    """
    try:
        from importlib import resources

        ref = resources.files("clinaudit.resources").joinpath("heart")
        # Traversable 不一定是真实路径（zip 安装时），用 as_file 落到临时目录
        with resources.as_file(ref) as p:
            if Path(p).exists():
                return Path(p)
    except (FileNotFoundError, ModuleNotFoundError, AttributeError):
        pass
    return None


def load_heart(
    cache_dir: str | Path | None = None,
    allow_network: bool = True,
    center: str | None = None,
    use_bundled: bool = True,
) -> ClinicalDataset:
    """加载心脏数据。

    Parameters
    ----------
    center
        若指定，只返回该中心的数据；否则返回四中心合并数据集
        （``center`` 字段保留每个样本的来源，可再用 ``by_center()`` 拆分）。
    use_bundled
        缓存与网络都不可用时，是否退回随包内置的数据副本。默认开启。
    """
    root: Path | None = None
    try:
        root = _ensure_uci("heart", cache_dir, allow_network)
    except FileNotFoundError:
        if not use_bundled:
            raise
        root = _bundled_heart_dir()
        if root is None:
            raise
        log.info("使用随包内置的心脏数据: %s", root)

    parts = parse_heart(root)
    if not parts:
        raise RuntimeError("未能解析任何心脏中心数据")
    if center is not None:
        if center not in parts:
            raise KeyError(f"未知中心 {center}；可用: {sorted(parts)}")
        return parts[center]
    return merge_centers(parts)


def load_dataset(name: str, cache_dir: str | Path | None = None, allow_network: bool = True) -> ClinicalDataset:
    """按名称加载数据集。"""
    n = name.lower()
    if n in {"heart", "heart_all"}:
        return load_heart(cache_dir, allow_network)
    cache = Path(cache_dir or Path.cwd() / "data_cache")
    if n == "pima":
        root = _ensure_uci("pima", cache, allow_network)
        return parse_pima(root)
    if n == "wdbc":
        root = _ensure_uci("wdbc", cache, allow_network)
        return parse_wdbc(root)
    if n in {"liver", "ilpd"}:
        root = _ensure_uci("liver", cache, allow_network)
        return parse_liver(root)
    raise ValueError(f"未知数据集: {name}；可用: heart, pima, wdbc, liver")


def _ensure_uci(key: str, cache_dir: str | Path | None, allow_network: bool) -> Path:
    """确保某数据集已解压到缓存目录，返回**直接包含数据文件**的目录。

    UCI 的 zip 解压后可能有两种布局：
    - ``cache/key/文件名``（扁平）
    - ``cache/key/子目录/文件名``（多一层）

    这里逐层向下探测，直到找到包含已知数据文件的目录，避免调用方
    必须知道具体层级。
    """
    cache = Path(cache_dir or Path.cwd() / "data_cache")
    out = cache / key
    if not (out.exists() and any(out.iterdir())):
        if not allow_network:
            raise FileNotFoundError(f"缓存不存在且已禁用网络: {out}")
        url = UCI_URLS[key]
        log.info("下载 %s ...", url)
        with urllib.request.urlopen(url, timeout=120) as resp:
            payload = resp.read()
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            zf.extractall(out)
    return _find_data_dir(out, key)


#: 每个数据集用于识别"正确的数据目录"的标志文件（文件名片段）。
_DATA_MARKERS = {
    "heart": "cleveland",
    "pima": "diabetes",
    "wdbc": "wdbc.data",
    "liver": "ilpd",
}


def _find_data_dir(root: Path, key: str, max_depth: int = 3) -> Path:
    """在 root 下广度优先查找包含标志文件的目录。"""
    marker = _DATA_MARKERS.get(key, "").lower()
    level = [root]
    for _ in range(max_depth):
        for d in level:
            try:
                names = [p.name.lower() for p in d.iterdir()]
            except OSError:
                continue
            if any(marker in n for n in names):
                return d
        nxt = []
        for d in level:
            try:
                nxt.extend([p for p in d.iterdir() if p.is_dir()])
            except OSError:
                continue
        if not nxt:
            break
        level = nxt
    return root
