"""审计流程编排：把数据、模型与各维度指标串成一份完整结论。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data.loaders import ClinicalDataset
from .metrics.abstention import abstention_summary
from .metrics.calibration import calibration_summary, net_benefit
from .metrics.discrimination import discrimination_summary
from .metrics.fairness import fairness_report, make_groups


@dataclass
class AuditResult:
    """一次审计的全部结果。"""

    dataset_name: str
    model_name: str
    n: int
    prevalence: float
    discrimination: dict
    calibration: dict
    fairness_center: dict | None
    fairness_sex: dict | None
    decision_curve: dict
    abstention: dict
    subgroup_table: list[dict] = field(default_factory=list)
    per_center: dict = field(default_factory=dict)
    score: float = float("nan")
    grade: str = ""
    flags: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        import numpy as np

        def clean(o):
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [clean(v) for v in o]
            if isinstance(o, (np.floating, float)):
                v = float(o)
                return None if not np.isfinite(v) else round(v, 6)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, np.ndarray):
                return clean(o.tolist())
            return o

        return clean(
            {
                "dataset": self.dataset_name,
                "model": self.model_name,
                "n": self.n,
                "prevalence": self.prevalence,
                "score": self.score,
                "grade": self.grade,
                "flags": self.flags,
                "discrimination": self.discrimination,
                "calibration": self.calibration,
                "fairness_center": self.fairness_center,
                "fairness_sex": self.fairness_sex,
                "decision_curve": {
                    k: v for k, v in self.decision_curve.items() if k != "thresholds"
                },
                "abstention": {
                    "base_risk": self.abstention["base_risk"],
                    "aurc": self.abstention["aurc"],
                    "at_coverage": self.abstention["at_coverage"],
                },
                "subgroup_table": self.subgroup_table,
                "per_center": self.per_center,
            }
        )


def _trust_score(disc: dict, cal: dict, fair: dict | None, ab: dict) -> tuple[float, str]:
    """把多维度指标综合成 0-100 的"可信度评分"与等级。

    这不是一个经过验证的临床量表，而是一个**用于自查的启发式汇总**，
    各分项的权重在下面显式给出，便于审查与调整。

    权重设计理由：
    - 校准（35）：临床决策直接依赖概率数值，校准差会导致系统性误治
    - 判别力（25）：AUC 仍是模型有用性的基础
    - 公平性（25）：亚组失效在临床上是严重安全事件
    - 拒识能力（15）：能否识别并交出不确定样本
    """
    parts: list[tuple[float, float]] = []  # (得分0-1, 权重)

    auc = disc.get("auc", np.nan)
    if np.isfinite(auc):
        # AUC 0.5 -> 0；0.9 及以上 -> 1
        parts.append((float(np.clip((auc - 0.5) / 0.4, 0, 1)), 0.25))

    ici = cal.get("ici", np.nan)
    if np.isfinite(ici):
        # ICI 0 -> 1；0.25 及以上 -> 0
        parts.append((float(np.clip(1 - ici / 0.25, 0, 1)), 0.35))

    if fair is not None and "diffs" in fair:
        a = fair["diffs"].get("auc_range", np.nan)
        if np.isfinite(a):
            parts.append((float(np.clip(1 - a / 0.25, 0, 1)), 0.25))

    if ab and np.isfinite(ab.get("aurc", np.nan)):
        base = ab.get("base_risk", np.nan)
        if np.isfinite(base) and base > 0:
            ratio = ab["aurc"] / base
            parts.append((float(np.clip(1 - (ratio - 0.5) / 0.5, 0, 1)), 0.15))

    if not parts:
        return float("nan"), "无法评分"
    wsum = sum(w for _, w in parts)
    score = 100.0 * sum(s * w for s, w in parts) / wsum

    if score >= 80:
        grade = "A（可信）"
    elif score >= 65:
        grade = "B（基本可用，需注意提示项）"
    elif score >= 50:
        grade = "C（存在明显缺陷，需改进后使用）"
    else:
        grade = "D（不建议直接用于临床决策）"
    return float(score), grade


def audit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    dataset_name: str = "unknown",
    model_name: str = "unknown",
    X: "object | None" = None,
    centers: np.ndarray | None = None,
    sex: np.ndarray | None = None,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> AuditResult:
    """执行完整审计。

    Parameters
    ----------
    X
        原始特征表；用于提取 ``sex`` 等敏感属性。可为 None。
    centers
        每个样本的中心标识，用于公平性分析。
    sex
        性别向量；为 None 时若 ``X`` 含 ``sex`` 列则自动提取。
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)

    disc = discrimination_summary(y, p, threshold=threshold)
    cal = calibration_summary(y, p, n_bins=n_bins)
    nb = net_benefit(y, p)
    ab = abstention_summary(y, p, threshold=threshold)

    fair_center = None
    if centers is not None and len(set(np.asarray(centers).tolist())) > 1:
        fair_center = fairness_report(y, p, np.asarray(centers, dtype=object), threshold)

    fair_sex = None
    sex_vec = sex
    if sex_vec is None and X is not None:
        try:
            if hasattr(X, "columns") and "sex" in X.columns:
                sex_vec = X["sex"].to_numpy()
        except Exception:  # noqa: BLE001
            sex_vec = None
    if sex_vec is not None:
        try:
            g, _names = make_groups(np.asarray(sex_vec, dtype=float))
            if len(set(g.tolist())) > 1:
                fair_sex = fairness_report(y, p, g, threshold)
        except Exception:  # noqa: BLE001
            fair_sex = None

    # 亚组表格
    table = []
    for fname, fr in (("中心", fair_center), ("性别", fair_sex)):
        if not fr:
            continue
        for gname, v in fr["groups"].items():
            if "note" in v:
                continue
            table.append(
                {
                    "attribute": fname,
                    "group": str(gname),
                    "n": v["n"],
                    "auc": v["auc"],
                    "sensitivity": v["sensitivity"],
                    "specificity": v["specificity"],
                    "calibration_gap": v["mean_gap"],
                }
            )

    flags: list[dict] = []
    if fair_center and "flags" in fair_center:
        flags.extend(fair_center["flags"])
    if fair_sex and "flags" in fair_sex:
        flags.extend(fair_sex["flags"])

    # 校准红旗
    if np.isfinite(cal.get("ici", np.nan)) and cal["ici"] > 0.10:
        flags.append(
            {
                "level": "高",
                "metric": "整体校准误差 ICI",
                "value": round(cal["ici"], 3),
                "message": "预测概率与实际发生率平均偏离超过 0.10，临床决策阈值不应直接采用",
            }
        )
    if np.isfinite(cal.get("slope", np.nan)) and cal["slope"] < 0.5:
        flags.append(
            {
                "level": "高",
                "metric": "校准斜率",
                "value": round(cal["slope"], 2),
                "message": "斜率显著小于 1，模型过度自信，预测概率分布过于极端",
            }
        )

    score, grade = _trust_score(disc, cal, fair_center, ab)

    return AuditResult(
        dataset_name=dataset_name,
        model_name=model_name,
        n=len(y),
        prevalence=float(y.mean()),
        discrimination=disc,
        calibration=cal,
        fairness_center=fair_center,
        fairness_sex=fair_sex,
        decision_curve=nb,
        abstention=ab,
        subgroup_table=table,
        score=score,
        grade=grade,
        flags=flags,
    )


def cross_center_audit(
    model,
    datasets: dict[str, ClinicalDataset],
    threshold: float = 0.5,
) -> dict:
    """多中心外部验证：在同一模型上逐个中心评估，输出漂移表。"""
    rows = []
    for name, ds in datasets.items():
        p = model.predict_proba(ds.X)
        disc = discrimination_summary(ds.y, p, threshold)
        cal = calibration_summary(ds.y, p)
        rows.append(
            {
                "center": name,
                "n": len(ds),
                "prevalence": round(ds.prevalence, 3),
                "missing": round(float(ds.X.isna().mean().mean()), 3),
                "auc": round(float(disc["auc"]), 4) if np.isfinite(disc["auc"]) else None,
                "ici": round(float(cal["ici"]), 4),
                "e50": round(float(cal["e50"]), 4),
                "e90": round(float(cal["e90"]), 4),
                "brier": round(float(cal["brier"]), 4),
                "slope": round(float(cal["slope"]), 3) if np.isfinite(cal["slope"]) else None,
                "mean_predicted": round(float(cal["mean_predicted"]), 4),
                "mean_gap": round(float(cal["mean_gap"]), 4),
            }
        )
    return {"centers": rows}
