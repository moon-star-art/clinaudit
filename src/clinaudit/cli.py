"""clinaudit 命令行界面。

用法
----
    clinaudit demo                              # 一键端到端演示
    clinaudit audit --data heart --model lr     # 单数据集审计
    clinaudit external --train cleveland        # 多中心外部验证
    clinaudit calibrate --data pima --method isotonic
    clinaudit datasets                          # 列出可用数据集
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _setup_matplotlib() -> None:
    if "MPLCONFIGDIR" not in os.environ:
        cache = Path.home() / ".cache" / "clinaudit" / "mpl"
        try:
            cache.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(cache)
        except OSError:
            pass


def _cache(args) -> Path:
    return Path(args.cache) if args.cache else Path.cwd() / "data_cache"


def _figures_for(res) -> dict:
    from .report import (
        plot_calibration,
        plot_decision_curve,
        plot_roc,
        plot_selective,
        plot_subgroup_bars,
    )

    figs = {
        "校准曲线": plot_calibration(res.calibration["calibration_curve"], n=res.n),
        "ROC 曲线": plot_roc(res.discrimination["roc_curve"], auc=res.discrimination["auc"]),
        "决策曲线": plot_decision_curve(res.decision_curve),
        "选择性预测": plot_selective(res.abstention["curve"]),
    }
    if res.fairness_center and "groups" in res.fairness_center:
        figs["各中心 AUC"] = plot_subgroup_bars(res.fairness_center["groups"], "auc", "各中心判别力")
    return figs


def _sections(res) -> list[dict]:
    cal = res.calibration
    disc = res.discrimination
    sections = [
        {
            "title": "判别力",
            "note": "AUC 只反映排序能力；PR-AUC 在类别不平衡时更有参考价值。",
            "table": {
                "columns": ["指标", "数值"],
                "rows": [
                    ["AUC", f"{disc['auc']:.4f}"],
                    ["PR-AUC", f"{disc['pr_auc']:.4f}"],
                    ["敏感度 (@0.5)", f"{disc['sensitivity']:.4f}"],
                    ["特异度 (@0.5)", f"{disc['specificity']:.4f}"],
                    ["阳性预测值", f"{disc['ppv']:.4f}"],
                    ["阴性预测值", f"{disc['npv']:.4f}"],
                    ["准确率", f"{disc['accuracy']:.4f}"],
                ],
            },
        },
        {
            "title": "校准",
            "note": "ICI 为校准曲线与对角线的加权平均偏差；E50/E90 为偏差的中位数与 90 分位数；"
            "斜率接近 1、截距接近 0 表示校准良好。",
            "table": {
                "columns": ["指标", "数值"],
                "rows": [
                    ["ICI", f"{cal['ici']:.4f}"],
                    ["E50", f"{cal['e50']:.4f}"],
                    ["E90", f"{cal['e90']:.4f}"],
                    ["平均偏差", f"{cal['mean_gap']:+.4f}"],
                    ["Brier", f"{cal['brier']:.4f}"],
                    ["校准斜率", f"{cal['slope']:.3f}"],
                    ["校准截距", f"{cal['intercept']:+.3f}"],
                    ["平均预测概率", f"{cal['mean_predicted']:.4f}"],
                    ["实际患病率", f"{cal['prevalence']:.4f}"],
                ],
            },
        },
        {
            "title": "临床效用（净收益）",
            "note": "净收益高于\"全部干预\"与\"全部不干预\"的阈值区间，才是模型真正有价值的区间。",
            "table": {
                "columns": ["阈值", "模型净收益", "全部干预", "模型更优"],
                "rows": [
                    [
                        f"{t:.2f}",
                        f"{m:+.4f}",
                        f"{a:+.4f}",
                        "是" if m > max(a, 0) else "否",
                    ]
                    for t, m, a in zip(
                        res.decision_curve["thresholds"][::10],
                        res.decision_curve["model"][::10],
                        res.decision_curve["treat_all"][::10],
                    )
                ],
            },
        },
        {
            "title": "拒识能力",
            "note": "AURC（风险-覆盖曲线下面积）越小，说明模型越能把错误集中在低置信度样本上。",
            "table": {
                "columns": ["目标覆盖率", "实际覆盖率", "错误率", "置信度阈值"],
                "rows": [
                    [
                        f"{r['target_coverage']:.2f}",
                        f"{r['achieved_coverage']:.2f}",
                        f"{r['risk']:.4f}",
                        f"{r['confidence_cutoff']:.3f}",
                    ]
                    for r in res.abstention["at_coverage"]
                ],
            },
        },
    ]
    if res.subgroup_table:
        sections.append(
            {
                "title": "亚组表现",
                "note": "成对比较各亚组的判别力、敏感度与校准偏差。",
                "table": {
                    "columns": ["属性", "组别", "样本数", "AUC", "敏感度", "特异度", "校准偏差"],
                    "rows": [
                        [
                            r["attribute"],
                            r["group"],
                            r["n"],
                            f"{r['auc']:.3f}",
                            f"{r['sensitivity']:.3f}",
                            f"{r['specificity']:.3f}",
                            f"{r['calibration_gap']:+.3f}",
                        ]
                        for r in res.subgroup_table
                    ],
                },
            }
        )
    return sections


def _conclusion(res) -> str:
    parts = [
        f"本次审计针对 <b>{res.dataset_name}</b> 数据集（n={res.n}，患病率 {res.prevalence:.1%}），"
        f"使用 <b>{res.model_name}</b> 模型。综合可信度评分 <b>{res.score:.1f}/100</b>，"
        f"等级 <b>{res.grade}</b>。"
    ]
    if res.flags:
        parts.append(
            f"共发现 <b>{len(res.flags)}</b> 项风险提示，其中高优先级 "
            f"{sum(1 for f in res.flags if f['level'] == '高')} 项。"
        )
    else:
        parts.append("未发现显著风险提示。")
    if res.fairness_center and "diffs" in res.fairness_center:
        a = res.fairness_center["diffs"].get("auc_range")
        if a is not None and a > 0.1:
            parts.append(
                f"各中心 AUC 差异达 <b>{a:.3f}</b>，说明模型在跨机构迁移时性能不稳定，"
                "部署前应在目标机构重新验证与校准。"
            )
    if res.calibration.get("ici", 0) > 0.1:
        parts.append(
            f"整体校准误差 ICI = <b>{res.calibration['ici']:.3f}</b>，"
            "预测概率不能直接作为临床风险解读，建议做概率重校准。"
        )
    return "".join(parts)


def cmd_demo(args) -> int:
    from .audit import audit, cross_center_audit
    from .data.loaders import load_heart
    from .models.training import train_model
    from .report import render_report

    print("=" * 64)
    print(" clinaudit 演示：多中心心脏数据审计")
    print("=" * 64)

    ds = load_heart(cache_dir=_cache(args), allow_network=args.allow_network)
    print(f"数据: {ds.summary()}")

    parts = ds.by_center()
    for k, v in parts.items():
        print(f"  {k:12s} n={len(v):4d} 患病率={v.prevalence:.3f} 缺失率={v.X.isna().mean().mean():.3f}")

    train = parts["cleveland"]
    print(f"\n训练模型 {args.model} on cleveland (n={len(train)}) ...")
    model = train_model(train.X, train.y, model=args.model, seed=0)

    print("\n各中心外部验证:")
    ext = cross_center_audit(model, parts)
    print(f"  {'中心':<13}{'n':>5}{'AUC':>8}{'ICI':>8}{'E50':>8}{'Brier':>8}{'斜率':>8}")
    for r in ext["centers"]:
        auc = f"{r['auc']:.3f}" if r["auc"] is not None else "  -  "
        slope = f"{r['slope']:.2f}" if r["slope"] is not None else " - "
        print(
            f"  {r['center']:<13}{r['n']:>5}{auc:>8}{r['ici']:>8.3f}"
            f"{r['e50']:>8.3f}{r['brier']:>8.3f}{slope:>8}"
        )

    p_all = model.predict_proba(ds.X)
    res = audit(
        ds.y, p_all,
        dataset_name="heart_all（四中心合并）",
        model_name=f"logistic 在 cleveland 上训练",
        X=ds.X, centers=ds.center,
    )
    res.per_center = ext["centers"]

    print(f"\n综合可信度: {res.score:.1f}/100  {res.grade}")
    for f in res.flags:
        print(f"  [{f['level']}] {f['metric']} = {f['value']}: {f['message']}")

    out = Path(args.out)
    figs = _figures_for(res)
    path = render_report(
        out,
        title="临床预测模型可信度审计报告",
        overview={
            "样本数": res.n,
            "患病率": f"{res.prevalence:.1%}",
            "AUC": f"{res.discrimination['auc']:.3f}",
            "ICI": f"{res.calibration['ici']:.3f}",
            "AURC": f"{res.abstention['aurc']:.3f}",
            "可信度": f"{res.score:.0f}",
        },
        sections=_sections(res),
        figures=figs,
        flags=res.flags,
        conclusion=_conclusion(res),
    )
    print(f"\n报告已生成: {path}")
    return 0


def cmd_audit(args) -> int:
    from .audit import audit
    from .data.loaders import load_dataset
    from .metrics.calibration import calibration_summary
    from .models.training import cross_val_predict_proba
    from .report import render_report

    ds = load_dataset(args.data, cache_dir=_cache(args), allow_network=args.allow_network)
    print(f"数据 {ds.name}: n={len(ds)} 特征={ds.n_features} 患病率={ds.prevalence:.3f}")

    p = cross_val_predict_proba(ds.X, ds.y, model=args.model, n_splits=args.cv, seed=0)
    res = audit(
        ds.y, p,
        dataset_name=ds.name,
        model_name=f"{args.model}（{args.cv} 折交叉验证）",
        X=ds.X,
        centers=ds.center,
    )
    print(f"AUC={res.discrimination['auc']:.4f}  ICI={res.calibration['ici']:.4f}  "
          f"评分={res.score:.1f} {res.grade}")
    for f in res.flags:
        print(f"  [{f['level']}] {f['metric']}={f['value']}: {f['message']}")

    out = Path(args.out)
    path = render_report(
        out,
        title=f"临床预测模型审计：{ds.name}",
        overview={
            "样本数": res.n,
            "特征数": ds.n_features,
            "AUC": f"{res.discrimination['auc']:.3f}",
            "ICI": f"{res.calibration['ici']:.3f}",
            "可信度": f"{res.score:.0f}",
        },
        sections=_sections(res),
        figures=_figures_for(res),
        flags=res.flags,
        conclusion=_conclusion(res),
    )
    print(f"报告: {path}")
    if args.json:
        Path(args.json).write_text(
            json.dumps(res.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"JSON: {args.json}")
    return 0


def cmd_external(args) -> int:
    from .audit import cross_center_audit
    from .data.loaders import load_heart
    from .models.training import train_model

    ds = load_heart(cache_dir=_cache(args), allow_network=args.allow_network)
    parts = ds.by_center()
    if args.train not in parts:
        print(f"未知训练中心 {args.train}；可用: {sorted(parts)}", file=sys.stderr)
        return 1

    tr = parts[args.train]
    print(f"训练中心: {args.train} (n={len(tr)}, 患病率={tr.prevalence:.3f})")
    model = train_model(tr.X, tr.y, model=args.model, seed=0)

    ext = cross_center_audit(model, parts)
    print(f"\n{'中心':<13}{'n':>5}{'患病率':>8}{'AUC':>8}{'ICI':>8}{'E50':>8}{'E90':>8}{'Brier':>8}{'斜率':>8}")
    print("-" * 76)
    for r in ext["centers"]:
        auc = f"{r['auc']:.3f}" if r["auc"] is not None else "  -  "
        slope = f"{r['slope']:>8.2f}" if r["slope"] is not None else "     -  "
        print(
            f"{r['center']:<13}{r['n']:>5}{r['prevalence']:>8.3f}{auc:>8}"
            f"{r['ici']:>8.3f}{r['e50']:>8.3f}{r['e90']:>8.3f}{r['brier']:>8.3f}{slope}"
        )

    if args.json:
        Path(args.json).write_text(
            json.dumps(ext, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nJSON: {args.json}")
    return 0


def cmd_calibrate(args) -> int:
    from .data.loaders import load_dataset
    from .metrics.calibration import calibration_summary, net_benefit
    from .models.training import cross_val_predict_proba, train_with_calibration
    from .models.training import train_model
    from sklearn.model_selection import train_test_split

    ds = load_dataset(args.data, cache_dir=_cache(args), allow_network=args.allow_network)
    Xtr, Xte, ytr, yte = train_test_split(
        ds.X, ds.y, test_size=0.4, random_state=0, stratify=ds.y
    )
    print(f"数据 {ds.name}: 训练 {len(ytr)} / 测试 {len(yte)}")

    raw = train_model(Xtr, ytr, model=args.model, seed=0)
    cal = train_with_calibration(Xtr, ytr, model=args.model, method=args.method, seed=0)

    print(f"\n{'方法':<16}{'ICI':>8}{'E50':>8}{'E90':>8}{'Brier':>8}{'斜率':>8}")
    print("-" * 56)
    for name, m in [(f"{args.model} 原始", raw), (f"+{args.method}", cal)]:
        p = m.predict_proba(Xte)
        c = calibration_summary(yte, p)
        print(f"{name:<16}{c['ici']:>8.3f}{c['e50']:>8.3f}{c['e90']:>8.3f}"
              f"{c['brier']:>8.3f}{c['slope']:>8.2f}")

    if args.json:
        out = {}
        for name, m in [("raw", raw), (args.method, cal)]:
            p = m.predict_proba(Xte)
            c = calibration_summary(yte, p)
            out[name] = {k: c[k] for k in ("ici", "e50", "e90", "brier", "slope", "intercept")}
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON: {args.json}")
    return 0


def cmd_datasets(args) -> int:
    from .data.loaders import HEART_CENTERS

    print("可用数据集:\n")
    print("  heart     UCI 心脏病（四中心合并，920 例，13 特征）")
    for k, (full, n) in HEART_CENTERS.items():
        print(f"              - {k:12s} {full}  (n={n})")
    print("  pima      Pima 印第安人糖尿病（768 例，8 特征）")
    print("  wdbc      威斯康星乳腺癌诊断（569 例，30 特征）")
    print("  liver     印度肝脏病人 ILPD（583 例，10 特征）")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clinaudit",
        description="临床预测模型可信度审计工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
        "  clinaudit demo\n"
        "  clinaudit audit --data heart --model lr\n"
        "  clinaudit external --train cleveland\n"
        "  clinaudit calibrate --data pima --method isotonic\n",
    )
    p.add_argument("--cache", default=None, help="数据缓存目录（默认 ./data_cache）")
    p.add_argument("--allow-network", action="store_true", help="允许联网下载数据")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="一键端到端演示")
    d.add_argument("--out", default="clinaudit_report.html")
    d.add_argument("--model", default="lr", choices=["lr", "gbdt", "rf"])
    d.set_defaults(func=cmd_demo)

    a = sub.add_parser("audit", help="单数据集审计")
    a.add_argument("--data", default="heart", help="数据集名（heart/pima/wdbc/liver）")
    a.add_argument("--model", default="lr", choices=["lr", "gbdt", "rf"])
    a.add_argument("--cv", type=int, default=5, help="交叉验证折数")
    a.add_argument("--out", default="audit_report.html")
    a.add_argument("--json", default=None)
    a.set_defaults(func=cmd_audit)

    e = sub.add_parser("external", help="多中心外部验证")
    e.add_argument("--train", default="cleveland", help="训练中心")
    e.add_argument("--model", default="lr", choices=["lr", "gbdt", "rf"])
    e.add_argument("--json", default=None)
    e.set_defaults(func=cmd_external)

    c = sub.add_parser("calibrate", help="重校准前后对比")
    c.add_argument("--data", default="pima")
    c.add_argument("--model", default="lr", choices=["lr", "gbdt", "rf"])
    c.add_argument("--method", default="sigmoid", choices=["sigmoid", "isotonic"])
    c.add_argument("--json", default=None)
    c.set_defaults(func=cmd_calibrate)

    ds = sub.add_parser("datasets", help="列出可用数据集")
    ds.set_defaults(func=cmd_datasets)
    return p


def main(argv: list[str] | None = None) -> int:
    _setup_matplotlib()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
