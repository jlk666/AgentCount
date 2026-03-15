"""Generate an HTML batch report with bar plots and pairwise t-tests."""

from __future__ import annotations

import base64
import io
import itertools
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats


# ── colour palette ────────────────────────────────────────────────────────────
_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
            "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

_CSS = """
:root {
  --bg: #f4f6f9;
  --card: #ffffff;
  --primary: #2d6a4f;
  --accent: #52b788;
  --text: #1b263b;
  --muted: #6c757d;
  --border: #dee2e6;
  --sig: #198754;
  --ns:  #dc3545;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: var(--bg);
       color: var(--text); font-size: 14px; }
header {
  background: linear-gradient(135deg, var(--primary) 0%, #1b4332 100%);
  color: #fff; padding: 28px 40px; display: flex; align-items: center; gap: 20px;
}
header img { height: 64px; border-radius: 8px; }
header h1  { font-size: 1.9rem; font-weight: 700; letter-spacing: .5px; }
header p   { font-size: .9rem; opacity: .8; margin-top: 4px; }
.container { max-width: 1200px; margin: 0 auto; padding: 32px 24px; }
h2 { font-size: 1.2rem; font-weight: 600; color: var(--primary);
     border-left: 4px solid var(--accent); padding-left: 10px; margin: 32px 0 14px; }
.card { background: var(--card); border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,.08); padding: 24px; margin-bottom: 24px; }
table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
thead tr { background: var(--primary); color: #fff; }
thead th { padding: 10px 14px; text-align: left; font-weight: 600; }
tbody tr:nth-child(even) { background: #f0f4f0; }
tbody td { padding: 9px 14px; border-bottom: 1px solid var(--border); }
.sig  { color: var(--sig); font-weight: 600; }
.ns   { color: var(--ns);  font-weight: 600; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.grid .card img { width: 100%; height: auto; }
.thumb-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }
.thumb-card { background: var(--card); border-radius: 8px;
              box-shadow: 0 1px 4px rgba(0,0,0,.08); overflow: hidden; }
.thumb-card img { width: 100%; display: block; }
.thumb-card .label { padding: 8px 12px; font-size: 12px; color: var(--muted); }
.thumb-card .label strong { display: block; font-size: 13px; color: var(--text); }
footer { text-align: center; padding: 24px; font-size: 12px; color: var(--muted); }
@media (max-width: 700px) { .grid { grid-template-columns: 1fr; } }
"""


# ── helpers ───────────────────────────────────────────────────────────────────

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _file_to_b64(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def _fmt_pval(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def _sig_class(p: float) -> str:
    return "sig" if p < 0.05 else "ns"


def _sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ── plotting ──────────────────────────────────────────────────────────────────

def _bar_plot(
    groups: dict[str, list[float]],
    ylabel: str,
    title: str,
    log_scale: bool = False,
) -> str:
    samples = list(groups.keys())
    means = [float(np.mean(v)) for v in groups.values()]
    sds   = [float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for v in groups.values()]

    fig, ax = plt.subplots(figsize=(max(5, len(samples) * 1.4), 4.5))
    colours = [_PALETTE[i % len(_PALETTE)] for i in range(len(samples))]
    x = np.arange(len(samples))
    bars = ax.bar(x, means, yerr=sds, capsize=5, width=0.55,
                  color=colours, edgecolor="white", linewidth=0.8,
                  error_kw=dict(elinewidth=1.2, ecolor="#444"))

    # scatter individual points
    for i, (key, vals) in enumerate(groups.items()):
        jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(vals))
        ax.scatter(x[i] + jitter, vals, color="white", edgecolors="#333",
                   s=40, zorder=3, linewidths=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(samples, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    if log_scale and all(m > 0 for m in means):
        ax.set_yscale("log")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── pairwise t-tests ──────────────────────────────────────────────────────────

def _pairwise_ttests(
    groups: dict[str, list[float]],
) -> list[dict[str, Any]]:
    results = []
    keys = list(groups.keys())
    for a, b in itertools.combinations(keys, 2):
        va, vb = groups[a], groups[b]
        if len(va) < 2 or len(vb) < 2:
            t_stat, p_val = float("nan"), float("nan")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                t_stat, p_val = stats.ttest_ind(va, vb, equal_var=False)
        results.append({
            "group_a": a,
            "group_b": b,
            "mean_a": float(np.mean(va)),
            "mean_b": float(np.mean(vb)),
            "n_a": len(va),
            "n_b": len(vb),
            "t_stat": t_stat,
            "p_value": p_val,
        })
    return results


# ── HTML builder ──────────────────────────────────────────────────────────────

def _ttest_table_html(rows: list[dict[str, Any]]) -> str:
    header = ("Group A", "Group B", "Mean A", "Mean B", "n A", "n B",
              "t-statistic", "p-value", "Significance")
    html = ['<table><thead><tr>']
    for h in header:
        html.append(f'<th>{h}</th>')
    html.append('</tr></thead><tbody>')
    for r in rows:
        p = r["p_value"]
        p_str = _fmt_pval(p) if not np.isnan(p) else "N/A"
        sig = _sig_label(p) if not np.isnan(p) else "N/A"
        css = _sig_class(p) if not np.isnan(p) else ""
        t_str = f"{r['t_stat']:.3f}" if not np.isnan(r["t_stat"]) else "N/A"
        html.append(
            f'<tr>'
            f'<td>{r["group_a"]}</td><td>{r["group_b"]}</td>'
            f'<td>{r["mean_a"]:,.2f}</td><td>{r["mean_b"]:,.2f}</td>'
            f'<td>{r["n_a"]}</td><td>{r["n_b"]}</td>'
            f'<td>{t_str}</td>'
            f'<td class="{css}">{p_str}</td>'
            f'<td class="{css}">{sig}</td>'
            f'</tr>'
        )
    html.append('</tbody></table>')
    return "\n".join(html)


def _summary_table_html(outputs: list[dict[str, Any]]) -> str:
    header = ("Sample", "Replicate", "Dilution", "Volume (mL)",
              "Colony Count", "CFU/mL", "QC Passed", "Validation Passed")
    html = ['<table><thead><tr>']
    for h in header:
        html.append(f'<th>{h}</th>')
    html.append('</tr></thead><tbody>')
    for state in outputs:
        md = state.get("metadata") or {}
        qc = (state.get("qc_status") or {}).get("passed", "—")
        val = (state.get("validation_status") or {}).get("passed", "—")
        cfu = state.get("cfu_per_ml")
        cfu_str = f"{cfu:,.0f}" if cfu is not None else "—"
        html.append(
            f'<tr>'
            f'<td>{md.get("sample_id","")}</td>'
            f'<td>{md.get("replicate_id","")}</td>'
            f'<td>{md.get("dilution","")}</td>'
            f'<td>{md.get("volume","")}</td>'
            f'<td>{state.get("colony_count","")}</td>'
            f'<td>{cfu_str}</td>'
            f'<td>{"✓" if qc is True else ("✗" if qc is False else "—")}</td>'
            f'<td>{"✓" if val is True else ("✗" if val is False else "—")}</td>'
            f'</tr>'
        )
    html.append('</tbody></table>')
    return "\n".join(html)


def _thumbnail_section(outputs: list[dict[str, Any]]) -> str:
    cards = []
    for state in outputs:
        md = state.get("metadata") or {}
        label = f'{md.get("sample_id","")} {md.get("replicate_id","")}'
        for key, title in [
            ("annotated_image_path", "Annotated"),
            ("sam_masks_image_path", "SAM Masks"),
        ]:
            path = state.get(key)
            if path and Path(path).exists():
                b64 = _file_to_b64(path)
                cards.append(
                    f'<div class="thumb-card">'
                    f'<img src="data:image/png;base64,{b64}" loading="lazy"/>'
                    f'<div class="label"><strong>{label}</strong>{title}</div>'
                    f'</div>'
                )
    return '<div class="thumb-grid">' + "\n".join(cards) + "</div>"


# ── public API ────────────────────────────────────────────────────────────────

def generate_batch_report(
    outputs: list[dict[str, Any]],
    report_path: Path,
    logo_path: Path | None = None,
    run_label: str = "Batch Run",
    timestamp: str = "",
) -> Path:
    """
    Build a self-contained HTML report from a completed batch run.

    Parameters
    ----------
    outputs     : list of workflow state dicts returned by run_batch()
    report_path : destination .html file path
    logo_path   : optional path to logo PNG (embedded as base64)
    run_label   : title string shown in the header
    timestamp   : ISO timestamp string for the header subtitle
    """
    # ── organise data by sample ──────────────────────────────────────────────
    count_groups: dict[str, list[float]] = {}
    cfu_groups:   dict[str, list[float]] = {}
    for state in outputs:
        md = state.get("metadata") or {}
        key = md.get("sample_id", "unknown")
        cnt = state.get("colony_count")
        cfu = state.get("cfu_per_ml")
        if cnt is not None:
            count_groups.setdefault(key, []).append(float(cnt))
        if cfu is not None:
            cfu_groups.setdefault(key, []).append(float(cfu))

    # ── plots ────────────────────────────────────────────────────────────────
    count_b64 = _bar_plot(count_groups, "Colony Count", "Colony Count per Sample")
    cfu_b64   = _bar_plot(cfu_groups,   "CFU/mL (log scale)", "CFU/mL per Sample",
                          log_scale=True)

    # ── statistics ───────────────────────────────────────────────────────────
    count_ttests = _pairwise_ttests(count_groups)
    cfu_ttests   = _pairwise_ttests(cfu_groups)

    # ── logo ─────────────────────────────────────────────────────────────────
    logo_tag = ""
    if logo_path and logo_path.exists():
        logo_b64 = _file_to_b64(logo_path)
        logo_tag = f'<img src="data:image/png;base64,{logo_b64}" alt="AgentCount logo"/>'

    subtitle = f"Generated: {timestamp}" if timestamp else ""

    # ── assemble HTML ────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AgentCount — {run_label}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  {logo_tag}
  <div>
    <h1>AgentCount — {run_label}</h1>
    <p>{subtitle}</p>
  </div>
</header>

<div class="container">

  <h2>Summary Results</h2>
  <div class="card">{_summary_table_html(outputs)}</div>

  <h2>Colony Count &amp; CFU/mL — Bar Plots</h2>
  <div class="grid">
    <div class="card">
      <img src="data:image/png;base64,{count_b64}" alt="Colony count bar plot"/>
    </div>
    <div class="card">
      <img src="data:image/png;base64,{cfu_b64}" alt="CFU/mL bar plot"/>
    </div>
  </div>

  <h2>Pairwise t-Tests — Colony Count</h2>
  <div class="card">
    <p style="font-size:12px;color:var(--muted);margin-bottom:10px;">
      Welch's two-sample t-test (unequal variance assumed).
      *** p &lt; 0.001 &nbsp; ** p &lt; 0.01 &nbsp; * p &lt; 0.05 &nbsp; ns = not significant.
    </p>
    {_ttest_table_html(count_ttests)}
  </div>

  <h2>Pairwise t-Tests — CFU/mL</h2>
  <div class="card">
    <p style="font-size:12px;color:var(--muted);margin-bottom:10px;">
      Welch's two-sample t-test (unequal variance assumed).
      *** p &lt; 0.001 &nbsp; ** p &lt; 0.01 &nbsp; * p &lt; 0.05 &nbsp; ns = not significant.
    </p>
    {_ttest_table_html(cfu_ttests)}
  </div>

  <h2>Output Images</h2>
  {_thumbnail_section(outputs)}

</div>
<footer>AgentCount &bull; Automated Colony Counting Pipeline</footer>
</body>
</html>"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    return report_path
