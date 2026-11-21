"""临床模型的训练与预测封装。

设计目标
--------
1. **可复现**：固定随机种子，交叉验证划分可重现。
2. **缺失值感知**：UCI 数据含较多缺失（Hungarian/Switzerland 尤甚），
   使用中位数填补 + 缺失指示列，避免静默丢弃样本。
3. **概率输出**：审计的核心是校准，因此必须拿到**概率**而非仅标签。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(
    model: str = "lr",
    seed: int = 0,
    add_missing_indicator: bool = True,
    scale: bool = True,
    n_jobs: int | None = None,
) -> Pipeline:
    """构造预处理 + 分类器的流水线。

    Parameters
    ----------
    model
        ``"lr"`` 逻辑回归 / ``"gbdt"`` 梯度提升树 / ``"rf"`` 随机森林。
    add_missing_indicator
        是否为每个含缺失的列追加"是否缺失"的指示特征。临床数据中
        "未做某项检查"本身携带信息（病情评估路径），不应被简单填补抹掉。
    n_jobs
        随机森林的并行度。``None`` 表示用 sklearn 默认值（单进程）。
        注意：在受限沙箱或某些 Windows 环境下，设为 ``-1``（全部核心）
        会因无法创建进程池而失败，因此默认不用多进程；需要时显式开启。
    """
    steps: list[tuple[str, object]] = []
    steps.append(
        (
            "impute",
            SimpleImputer(
                strategy="median",
                add_indicator=add_missing_indicator,
                keep_empty_features=True,
            ),
        )
    )

    m = model.lower()
    if m in {"lr", "logistic"}:
        if scale:
            steps.append(("scale", StandardScaler()))
        steps.append(
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            )
        )
    elif m in {"gbdt", "tree"}:
        steps.append(
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=120,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=seed,
                ),
            )
        )
    elif m in {"rf", "forest"}:
        steps.append(
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=n_jobs,
                ),
            )
        )
    else:
        raise ValueError(f"未知模型: {model}；可用: lr, gbdt, rf")

    return Pipeline(steps)


@dataclass
class TrainedModel:
    """训练好的模型及其预测结果。"""

    pipeline: Pipeline
    model_name: str
    feature_names: list[str] = field(default_factory=list)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model: str = "lr",
    seed: int = 0,
) -> TrainedModel:
    """在训练集上拟合流水线。"""
    pipe = build_pipeline(model=model, seed=seed)
    pipe.fit(X_train, y_train)
    return TrainedModel(pipeline=pipe, model_name=model, feature_names=list(X_train.columns))


def cross_val_predict_proba(
    X: pd.DataFrame,
    y: np.ndarray,
    model: str = "lr",
    n_splits: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """交叉验证的袋外预测概率。

    用于在没有独立验证集时估计模型的泛化表现。分层 K 折保证每折的
    阳性比例与总体一致。
    """
    n_splits = min(n_splits, int(np.bincount(y).min()))
    if n_splits < 2:
        raise ValueError(
            f"阳性或阴性样本过少（每类 {np.bincount(y).min()} 例），无法做交叉验证"
        )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X, y):
        pipe = build_pipeline(model=model, seed=seed)
        pipe.fit(X.iloc[tr], y[tr])
        out[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    return out


def train_with_calibration(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model: str = "lr",
    method: str = "sigmoid",
    n_splits: int = 5,
    seed: int = 0,
) -> TrainedModel:
    """训练后做概率重校准（Platt / Isotonic）。

    Parameters
    ----------
    method
        ``"sigmoid"``（Platt scaling）或 ``"isotonic"``（保序回归）。
        保序回归更灵活但需要更多数据，小样本时易过拟合。
    """
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("method 必须是 'sigmoid' 或 'isotonic'")
    base = build_pipeline(model=model, seed=seed)
    n_splits = min(n_splits, int(np.bincount(y_train).min()))
    if n_splits < 2:
        raise ValueError("样本过少，无法做校准")
    cal = CalibratedClassifierCV(base, method=method, cv=n_splits)
    cal.fit(X_train, y_train)
    return TrainedModel(pipeline=cal, model_name=f"{model}+{method}", feature_names=list(X_train.columns))
