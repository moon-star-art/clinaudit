"""clinaudit 的图表与中文 HTML 审计报告。"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ------------------------------------------------------------------ 单图
def plot_calibration(curve_bins: list[dict], title: str = "校准曲线", n: int | None = None) -> str:
    """校准曲线：预测概率 vs 实际发生率，附对角线参考。"""
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="完美校准")
    if curve_bins:
        xs = [b["mean_predicted"] for b in curve_bins]
        ys = [b["fraction_positive"] for b in curve_bins]
        ns = np.array([b["n"] for b in curve_bins], dtype=float)
        ax.plot(xs, ys, "o-", color="#0f766e", lw=1.6, ms=4, label="模型")
        # 用气泡大小表达每箱样本量
        ax.scatter(xs, ys, s=30 + 200 * (ns / ns.max()), color="#0f766e", alpha=0.25, zorder=2)
    ax.set_xlabel("预测概率", fontsize=9)
    ax.set_ylabel("实际阳性率", fontsize=9)
    ax.set_title(title + (f" (n={n})" if n else ""), fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _b64(fig)


def plot_roc(roc: dict, title: str = "ROC 曲线", auc: float | None = None) -> str:
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot(roc["fpr"], roc["tpr"], color="#0f766e", lw=1.6)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("假阳性率", fontsize=9)
    ax.set_ylabel("真阳性率", fontsize=9)
    t = title + (f"  AUC={auc:.3f}" if auc is not None and np.isfinite(auc) else "")
    ax.set_title(t, fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _b64(fig)


def plot_decision_curve(nb: dict, title: str = "决策曲线") -> str:
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    th = np.array(nb["thresholds"])
    ax.plot(th, nb["model"], color="#0f766e", lw=1.8, label="模型")
    ax.plot(th, nb["treat_all"], color="#b91c1c", lw=1.2, ls="--", label="全部干预")
    ax.axhline(0, color="gray", lw=1.2, ls=":", label="全部不干预")
    ax.set_xlabel("阈值概率", fontsize=9)
    ax.set_ylabel("净收益", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _b64(fig)


def plot_selective(curve: dict, title: str = "选择性预测") -> str:
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    ax.plot(curve["coverage"], curve["risk"], color="#0f766e", lw=1.8)
    ax.axhline(curve["base_risk"], color="#b91c1c", ls="--", lw=1.2,
               label=f"不拒识 (风险={curve['base_risk']:.3f})")
    ax.set_xlabel("覆盖率", fontsize=9)
    ax.set_ylabel("错误率", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _b64(fig)


def plot_subgroup_bars(groups: dict, metric: str = "auc", title: str = "") -> str:
    """亚组指标条形图。"""
    keys, vals = [], []
    for k, v in groups.items():
        if metric in v and np.isfinite(v[metric]):
            keys.append(k)
            vals.append(v[metric])
    fig, ax = plt.subplots(figsize=(max(3.4, 1.1 * len(keys) + 1.4), 2.8))
    if keys:
        colors = ["#0f766e" if v == max(vals) else "#94a3b8" for v in vals]
        ax.bar(keys, vals, color=colors)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
        lo, hi = min(vals), max(vals)
        ax.set_ylim(max(0, lo - 0.15), min(1.05, hi + 0.1))
    ax.set_ylabel(metric.upper(), fontsize=9)
    ax.set_title(title or f"亚组 {metric}", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _b64(fig)


# ------------------------------------------------------------------ 报告
def render_report(
    out_path: str | Path,
    title: str,
    overview: dict,
    sections: list[dict],
    figures: dict[str, str] | None = None,
    flags: list[dict] | None = None,
    conclusion: str = "",
) -> Path:
    """渲染中文 HTML 审计报告。

    Parameters
    ----------
    overview
        顶部 KPI 卡片，``{标签: 值}``。
    sections
        每个元素形如 ``{"title": str, "table": {"columns": [...], "rows": [[...]]},
        "note": str}``。
    flags
        红旗列表，每项含 ``level`` / ``metric`` / ``value`` / ``message``。
    """
    from jinja2 import Template

    html = Template(_TEMPLATE).render(
        title=title,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        overview=overview,
        sections=sections,
        figures=figures or {},
        flags=flags or [],
        conclusion=conclusion,
    )
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{{ title }}</title>
<style>
  :root { --fg:#1f2937; --muted:#6b7280; --line:#e5e7eb; --accent:#0f766e;
          --warn:#b45309; --bad:#b91c1c; --bg:#fbfbfa; }
  * { box-sizing:border-box; }
  body { font-family:"Microsoft YaHei","PingFang SC","Noto Sans SC",system-ui,sans-serif;
         color:var(--fg); background:var(--bg); margin:0; padding:32px 24px; line-height:1.7; }
  .wrap { max-width:1000px; margin:0 auto; }
  h1 { font-size:22px; border-bottom:2px solid var(--accent); padding-bottom:10px; }
  h2 { font-size:16px; margin-top:34px; color:var(--accent);
       border-left:4px solid var(--accent); padding-left:10px; }
  .meta { color:var(--muted); font-size:12px; margin-bottom:20px; }
  .kpis { display:flex; gap:12px; flex-wrap:wrap; margin:16px 0; }
  .kpi { flex:1 1 140px; background:#fff; border:1px solid var(--line);
         border-radius:8px; padding:12px 14px; }
  .kpi .v { font-size:20px; font-weight:700; color:var(--accent); }
  .kpi .l { font-size:12px; color:var(--muted); }
  table { width:100%; border-collapse:collapse; font-size:13px; margin:12px 0; background:#fff; }
  th,td { border:1px solid var(--line); padding:7px 10px; text-align:right; }
  th:first-child, td:first-child { text-align:left; }
  thead th { background:#f3f4f6; font-weight:600; }
  tbody tr:nth-child(even) { background:#fafafa; }
  .figs { display:flex; flex-wrap:wrap; gap:14px; margin:14px 0; }
  .figs figure { margin:0; background:#fff; border:1px solid var(--line);
                 border-radius:8px; padding:8px; }
  .figs img { max-width:100%; display:block; }
  .figs figcaption { font-size:11px; color:var(--muted); text-align:center; margin-top:4px; }
  .flag { border-left:4px solid var(--warn); background:#fffbeb; padding:10px 14px;
          margin:10px 0; border-radius:4px; font-size:13px; }
  .flag.high { border-left-color:var(--bad); background:#fef2f2; }
  .flag .lv { font-weight:700; }
  .note { font-size:12px; color:var(--muted); }
  .concl { background:#fff; border:1px solid var(--line); border-radius:8px;
           padding:16px; margin:16px 0; font-size:13.5px; }
  footer { margin-top:44px; font-size:11px; color:var(--muted);
           border-top:1px solid var(--line); padding-top:12px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>{{ title }}</h1>
  <div class="meta">生成时间：{{ generated }} ｜ clinaudit</div>

  <h2>一、概览</h2>
  <div class="kpis">
  {% for k, v in overview.items() %}
    <div class="kpi"><div class="v">{{ v }}</div><div class="l">{{ k }}</div></div>
  {% endfor %}
  </div>

  {% if flags %}
  <h2>二、风险提示</h2>
  {% for f in flags %}
  <div class="flag {{ 'high' if f.level == '高' else '' }}">
    <span class="lv">[{{ f.level }}] {{ f.metric }}</span>（{{ f.value }}）：{{ f.message }}
  </div>
  {% endfor %}
  {% endif %}

  {% for s in sections %}
  <h2>{{ s.title }}</h2>
  {% if s.note %}<p class="note">{{ s.note }}</p>{% endif %}
  {% if s.table %}
  <table>
    <thead><tr>{% for c in s.table.columns %}<th>{{ c }}</th>{% endfor %}</tr></thead>
    <tbody>
    {% for r in s.table.rows %}
      <tr>{% for c in r %}<td>{{ c }}</td>{% endfor %}</tr>
    {% endfor %}
    </tbody>
  </table>
  {% endif %}
  {% endfor %}

  {% if figures %}
  <h2>图表</h2>
  <div class="figs">
  {% for name, b64 in figures.items() %}
    <figure><img src="data:image/png;base64,{{ b64 }}"><figcaption>{{ name }}</figcaption></figure>
  {% endfor %}
  </div>
  {% endif %}

  {% if conclusion %}
  <h2>结论</h2>
  <div class="concl">{{ conclusion }}</div>
  {% endif %}

  <footer>由 clinaudit 自动生成。所有数值均来自本次运行，未做人工调整。</footer>
</div>
</body>
</html>
"""
