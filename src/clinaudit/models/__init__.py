"""clinaudit.models —— 临床模型的训练与校准。"""

from __future__ import annotations

from .training import (
    TrainedModel,
    build_pipeline,
    cross_val_predict_proba,
    train_model,
    train_with_calibration,
)

__all__ = [
    "TrainedModel",
    "build_pipeline",
    "cross_val_predict_proba",
    "train_model",
    "train_with_calibration",
]
