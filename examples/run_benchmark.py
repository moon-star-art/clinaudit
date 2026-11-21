"""clinaudit 基准实验：生成论文所需的全部结果表与图形。

运行：
    python examples/run_benchmark.py --out results

产物（results/ 下）：
    benchmark.json          全部数值结果
    external.csv            多中心外部验证表
    calibration.csv         重校准前后对比
    fig_*.png               论文用图
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinaudit.audit import audit, cross_center_audit
from clinaudit.data.loaders import load_heart
from clinaudit.metrics.abstention import (
    abstention_summary,
    conformal_prediction_sets,
    empirical_coverage,
)
from clinaudit.metrics.calibration import calibration_summary, net_benefit
from clinaudit.metrics.discrimination import discrimination_summary, roc_curve_points
from clinaudit.metrics.fairness import fairness_report, make_groups
from clinaudit.models.training import (
    cross_val_predict_proba,
    train_model,
    train_with_calibration,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130


def main() -> int:
    ap = argparse.ArgumentParser(description="clinaudit 基准实验")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--allow-network", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = args.cache or (Path(__file__).resolve().parents[1] / "data_cache")

    # ------------------------------------------------------------ 数据
    ds = load_heart(cache_dir=cache, allow_network=args.allow_network)
    parts = ds.by_center()
    print(f"[数据] 合并 {len(ds)} 例，{len(parts)} 个中心")
    for k, v in parts.items():
        print(f"  {k:12s} n={len(v):4d} 患病率={v.prevalence:.3f} "
              f"缺失率={v.X.isna().mean().mean():.3f}")

    # ------------------------------------------- 实验1：多中心外部验证
    print("\n[实验1] 多中心外部验证（Cleveland 训练）")
    tr = parts["cleveland"]
    model = train_model(tr.X, tr.y, model="lr", seed=0)
    ext = cross_center_audit(model, parts)
    print(f"  {'中心':<13}{'n':>5}{'AUC':>8}{'ICI':>8}{'E50':>8}{'Brier':>8}{'斜率':>8}")
    for r in ext["centers"]:
        auc = f"{r['auc']:.3f}" if r["auc"] is not None else "  -  "
        slope = f"{r['slope']:>8.2f}" if r["slope"] is not None else "     -  "
        print(f"  {r['center']:<13}{r['n']:>5}{auc:>8}{r['ici']:>8.3f}"
              f"{r['e50']:>8.3f}{r['brier']:>8.3f}{slope}")

    with (out / "external.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ext["centers"][0].keys()))
        w.writeheader()
        w.writerows(ext["centers"])

    # --------------------------------------- 实验2：重校准效果对比
    print("\n[实验2] 概率重校准（Cleveland 训练 -> Switzerland 验证）")
    from sklearn.model_selection import train_test_split

    Xc, Xc_te, yc, yc_te = train_test_split(
        tr.X, tr.y, test_size=0.4, random_state=0, stratify=tr.y
    )
    sw = parts["switzerland"]
    recal_rows = []
    raw_m = train_model(Xc, yc, model="lr", seed=0)
    print(f"  {'方法':<16}{'ICI':>8}{'E50':>8}{'Brier':>8}{'斜率':>8}")
    for name, m in [
        ("原始 logistic", raw_m),
        ("+Platt", train_with_calibration(Xc, yc, model="lr", method="sigmoid", seed=0)),
        ("+Isotonic", train_with_calibration(Xc, yc, model="lr", method="isotonic", seed=0)),
    ]:
        p = m.predict_proba(sw.X)
        c = calibration_summary(sw.y, p)
        recal_rows.append(
            {
                "method": name,
                "ici": round(float(c["ici"]), 4),
                "e50": round(float(c["e50"]), 4),
                "e90": round(float(c["e90"]), 4),
                "brier": round(float(c["brier"]), 4),
                "slope": round(float(c["slope"]), 3),
                "intercept": round(float(c["intercept"]), 3),
            }
        )
        print(f"  {name:<16}{c['ici']:>8.3f}{c['e50']:>8.3f}{c['brier']:>8.3f}{c['slope']:>8.2f}")

    with (out / "calibration.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recal_rows[0].keys()))
        w.writeheader()
        w.writerows(recal_rows)

    # --------------------------------------------- 实验3：整体审计
    print("\n[实验3] 整体审计（四中心合并）")
    p_all = model.predict_proba(ds.X)
    res = audit(
        ds.y, p_all,
        dataset_name="heart_all",
        model_name="logistic@cleveland",
        X=ds.X, centers=ds.center,
    )
    print(f"  AUC={res.discrimination['auc']:.3f}  ICI={res.calibration['ici']:.3f}  "
          f"评分={res.score:.1f}  {res.grade}")
    print(f"  亚组差异: AUC range={res.fairness_center['diffs']['auc_range']:.3f}, "
          f"敏感度 range={res.fairness_center['diffs']['sensitivity_range']:.3f}")
    for f in res.flags:
        print(f"  [{f['level']}] {f['metric']}={f['value']}: {f['message']}")

    # ------------------------------------------ 实验4：Conformal 覆盖率
    print("\n[实验4] Conformal 预测的经验覆盖率")
    conf_rows = []
    cs = parts["cleveland"]
    Xc2, Xc_t, yc2, yc_t = train_test_split(
        cs.X, cs.y, test_size=0.5, random_state=1, stratify=cs.y
    )
    Xca, Xcal, yca, ycal = train_test_split(
        Xc2, yc2, test_size=0.5, random_state=1, stratify=yc2
    )
    m2 = train_model(Xca, yca, model="lr", seed=0)
    p_cal = m2.predict_proba(Xcal)
    p_test = m2.predict_proba(Xc_t)
    for alpha in (0.05, 0.10, 0.20):
        sets = conformal_prediction_sets(ycal, p_cal, p_test, alpha=alpha)
        cov = empirical_coverage(yc_t, sets)
        conf_rows.append(
            {
                "alpha": alpha,
                "target": 1 - alpha,
                "empirical": round(cov, 4),
                "mean_set_size": round(float(sets["mean_set_size"]), 3),
                "empty_fraction": round(float(sets["empty_set_fraction"]), 4),
            }
        )
        print(f"  alpha={alpha:.2f}  目标={1-alpha:.2f}  实际={cov:.4f}  "
              f"平均集合大小={sets['mean_set_size']:.2f}")

    # -------------------------------- 实验5：不同模型的拒识与校准
    print("\n[实验5] 不同分类器的校准与拒识")
    model_rows = []
    for mname in ("lr", "gbdt", "rf"):
        p = cross_val_predict_proba(ds.X, ds.y, model=mname, n_splits=5, seed=0)
        d = discrimination_summary(ds.y, p)
        c = calibration_summary(ds.y, p)
        ab = abstention_summary(ds.y, p)
        model_rows.append(
            {
                "model": mname,
                "auc": round(float(d["auc"]), 4),
                "pr_auc": round(float(d["pr_auc"]), 4),
                "ici": round(float(c["ici"]), 4),
                "e50": round(float(c["e50"]), 4),
                "brier": round(float(c["brier"]), 4),
                "slope": round(float(c["slope"]), 3),
                "aurc": round(float(ab["aurc"]), 4),
            }
        )
        print(f"  {mname:6s} AUC={d['auc']:.3f} ICI={c['ici']:.3f} "
              f"Brier={c['brier']:.3f} 斜率={c['slope']:.2f} AURC={ab['aurc']:.3f}")

    # 按性别公平性
    sex_g, _ = make_groups(ds.X["sex"].fillna(-1).to_numpy())
    fair_sex = fairness_report(ds.y, p_all, sex_g)
    print(f"\n  按性别: AUC range={fair_sex['diffs']['auc_range']:.3f}, "
          f"敏感度 range={fair_sex['diffs']['sensitivity_range']:.3f}")

    # ------------------------------------------------------------ 汇总
    result = {
        "dataset": ds.summary(),
        "external_validation": ext,
        "recalibration": recal_rows,
        "overall_audit": res.to_dict(),
        "conformal": conf_rows,
        "models": model_rows,
        "fairness_sex": {
            "groups": fair_sex["groups"],
            "diffs": fair_sex["diffs"],
            "flags": fair_sex["flags"],
        },
    }
    (out / "benchmark.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ------------------------------------------------------------ 绘图
    # 图1：各中心校准曲线
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (cname, cds) in zip(axes, parts.items()):
        p = model.predict_proba(cds.X)
        c = calibration_summary(cds.y, p)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        xs = [b["mean_predicted"] for b in c["calibration_curve"]]
        ys = [b["fraction_positive"] for b in c["calibration_curve"]]
        ax.plot(xs, ys, "o-", color="#0f766e", ms=3, lw=1.4)
        ax.set_title(f"{cname}\nICI={c['ici']:.3f}", fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("实际阳性率", fontsize=8)
    for ax in axes:
        ax.set_xlabel("预测概率", fontsize=8)
    fig.suptitle("图1  多中心外部验证的校准曲线（同一模型，不同中心）", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig1_calibration_centers.png")
    plt.close(fig)

    # 图2：各中心 AUC / ICI 对比
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    names = [r["center"] for r in ext["centers"]]
    aucs = [r["auc"] if r["auc"] is not None else 0 for r in ext["centers"]]
    icis = [r["ici"] for r in ext["centers"]]
    axes[0].bar(names, aucs, color=["#0f766e" if a == max(aucs) else "#94a3b8" for a in aucs])
    axes[0].set_ylabel("AUC")
    axes[0].set_title("判别力（AUC）", fontsize=10)
    axes[0].axhline(0.5, ls="--", c="gray", lw=1)
    axes[1].bar(names, icis, color=["#b91c1c" if i > 0.1 else "#94a3b8" for i in icis])
    axes[1].set_ylabel("ICI")
    axes[1].set_title("校准误差（ICI，越小越好）", fontsize=10)
    axes[1].axhline(0.1, ls="--", c="#b45309", lw=1, label="0.10 警戒线")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(labelsize=8)
    fig.suptitle("图2  同一模型在四个中心的判别力与校准", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig2_centers_auc_ici.png")
    plt.close(fig)

    # 图3：决策曲线
    nb = net_benefit(ds.y, p_all)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(nb["thresholds"], nb["model"], color="#0f766e", lw=1.8, label="模型")
    ax.plot(nb["thresholds"], nb["treat_all"], color="#b91c1c", ls="--", lw=1.2, label="全部干预")
    ax.axhline(0, color="gray", ls=":", lw=1.2, label="全部不干预")
    ax.set_xlabel("阈值概率")
    ax.set_ylabel("净收益")
    ax.set_title("图3  决策曲线分析")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig3_decision_curve.png")
    plt.close(fig)

    # 图4：选择性预测
    ab = abstention_summary(ds.y, p_all)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(ab["curve"]["coverage"], ab["curve"]["risk"], color="#0f766e", lw=1.8)
    ax.axhline(ab["base_risk"], color="#b91c1c", ls="--", lw=1.2,
               label=f"不拒识 风险={ab['base_risk']:.3f}")
    ax.set_xlabel("覆盖率")
    ax.set_ylabel("错误率")
    ax.set_title(f"图4  选择性预测（AURC={ab['aurc']:.3f}）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig4_selective.png")
    plt.close(fig)

    print(f"\n[完成] 结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
