"""Plotly figure helpers for Stage 00 validation diagnostics.

All figures are written in two forms:

- Interactive HTML at `<figures_dir>/<name>.html`
- Static PNG at `<figures_dir>/<name>.png` (via `kaleido`)

The PNG form is intended for the LaTeX report; the HTML form is intended
for the GitHub-rendered notebook.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

log = logging.getLogger(__name__)


def save_figure(
    fig: go.Figure,
    name: str,
    figures_dir: Path,
    *,
    width: int = 900,
    height: int = 560,
    write_html: bool = True,
    write_png: bool = True,
) -> dict[str, Path]:
    """Write a Plotly figure to HTML and PNG side-by-side.

    Returns the paths actually written.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if write_html:
        html_path = figures_dir / f"{name}.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
        written["html"] = html_path

    if write_png:
        png_path = figures_dir / f"{name}.png"
        try:
            fig.write_image(str(png_path), format="png", width=width, height=height)
            written["png"] = png_path
        except Exception as exc:  # noqa: BLE001 — kaleido import / env issues
            log.warning("kaleido PNG export failed for %s: %s", name, exc)

    log.info("figure %s → %s", name, {k: str(v) for k, v in written.items()})
    return written


def coverage_stability_figure(panel: pd.DataFrame) -> go.Figure:
    """Monthly firm count and YoY change time-series."""
    counts = (
        panel.groupby(panel["date"].dt.to_period("M"))["permno"]
        .nunique()
        .rename("n_firms")
    )
    counts.index = counts.index.to_timestamp("M") + pd.offsets.MonthEnd(0)
    yoy = counts.pct_change(12).rename("yoy")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=counts.index,
            y=counts.values,
            mode="lines",
            name="N_t (firms)",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=yoy.index,
            y=yoy.values,
            mode="lines",
            name="YoY change",
            yaxis="y2",
            line=dict(dash="dot"),
        )
    )
    fig.update_layout(
        title="Stage 00 panel coverage (firm count and YoY change)",
        xaxis_title="Month",
        yaxis=dict(title="Firms in panel"),
        yaxis2=dict(title="YoY change", overlaying="y", side="right", tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=60, r=60, t=60, b=40),
    )
    return fig


def char_presence_figure(diag: pd.DataFrame) -> go.Figure:
    """Bar chart of pre-filter characteristic presence rates."""
    diag = diag.sort_values("pct_present").reset_index(drop=True)
    fig = go.Figure(
        go.Bar(
            x=diag["pct_present"] * 100.0,
            y=diag["characteristic"],
            orientation="h",
            marker=dict(color=diag["pct_present"], colorscale="Viridis"),
        )
    )
    fig.update_layout(
        title="Pre-filter characteristic presence (rows non-null)",
        xaxis_title="Percent of rows non-null",
        yaxis_title="Characteristic",
        height=900,
        margin=dict(l=120, r=40, t=60, b=40),
    )
    return fig


# -----------------------------------------------------------------------------
# Arm 1 distributional comparison figures
# -----------------------------------------------------------------------------


def arm1_annual_breadth_figure(breadth: pd.DataFrame) -> go.Figure:
    """Two lines: ours and CPZ yearly mean firms-per-month."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=breadth["year"],
            y=breadth["ours_breadth"],
            mode="lines+markers",
            name="Ours",
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=breadth["year"],
            y=breadth["cpz_breadth"],
            mode="lines+markers",
            name="CPZ",
            line=dict(width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title="Arm 1 D1: Annual breadth (firms per month) --- Ours vs CPZ",
        xaxis_title="Year",
        yaxis_title="Average stocks per month",
        legend=dict(x=0.02, y=0.98),
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=40),
    )
    return fig


def arm1_annual_returns_figure(returns: pd.DataFrame) -> go.Figure:
    """Two lines: ours and CPZ yearly equal-weighted excess return."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=returns["year"],
            y=returns["ours_ann_ret"],
            mode="lines+markers",
            name="Ours",
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=returns["year"],
            y=returns["cpz_ann_ret"],
            mode="lines+markers",
            name="CPZ",
            line=dict(width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title="Arm 1 D2: Annual equal-weighted excess return --- Ours vs CPZ",
        xaxis_title="Year",
        yaxis_title="Annual mean excess return",
        yaxis_tickformat=".2%",
        legend=dict(x=0.02, y=0.98),
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=40),
    )
    return fig


def arm1_char_stats_figure(stats: pd.DataFrame) -> go.Figure:
    """Two horizontal-bar panels: |diff_mean| and |diff_std| per characteristic."""
    stats = stats.copy()
    stats["abs_diff_mean"] = stats["diff_mean"].abs()
    stats["abs_diff_std"] = stats["diff_std"].abs()
    stats = stats.sort_values("abs_diff_mean")

    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("|Ours mean - CPZ mean|", "|Ours std - CPZ std|"),
        shared_yaxes=True,
    )
    fig.add_trace(
        go.Bar(
            x=stats["abs_diff_mean"],
            y=stats["characteristic"],
            orientation="h",
            name="|diff mean|",
            marker=dict(color=stats["abs_diff_mean"], colorscale="Viridis"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=stats["abs_diff_std"],
            y=stats["characteristic"],
            orientation="h",
            name="|diff std|",
            marker=dict(color=stats["abs_diff_std"], colorscale="Viridis"),
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title="Arm 1 D3: Per-characteristic full-period stat differences (rank-norm scale)",
        showlegend=False,
        template="plotly_white",
        height=900,
        margin=dict(l=120, r=40, t=80, b=40),
    )
    return fig


def arm1_yearly_coverage_figure(coverage: pd.DataFrame) -> go.Figure:
    """Heatmap of per-(year, char) pre-filter coverage."""
    wide = coverage.pivot_table(index="characteristic", columns="year", values="coverage")
    fig = go.Figure(
        data=go.Heatmap(
            z=wide.values,
            x=wide.columns,
            y=wide.index,
            colorscale="Viridis",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Coverage"),
        )
    )
    fig.update_layout(
        title="Arm 1 D4: Pre-filter characteristic coverage by year (our panel)",
        xaxis_title="Year",
        yaxis_title="Characteristic",
        template="plotly_white",
        height=900,
        margin=dict(l=120, r=40, t=60, b=40),
    )
    return fig


# -----------------------------------------------------------------------------
# Arm 2 figures
# -----------------------------------------------------------------------------


def arm2_cumulative_returns_figure(joined: pd.DataFrame) -> go.Figure:
    """6-panel grid (2x3) of cumulative factor returns: Ours vs Ken French."""
    from plotly.subplots import make_subplots

    pairs = [
        ("mkt_ours", "mktrf", "MKT-RF"),
        ("smb_ours", "smb", "SMB"),
        ("hml_ours", "hml", "HML"),
        ("rmw_ours", "rmw", "RMW"),
        ("cma_ours", "cma", "CMA"),
        ("umd_ours", "umd", "UMD"),
    ]
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[label for _, _, label in pairs],
        shared_xaxes=True,
    )
    for idx, (ours_col, kf_col, _label) in enumerate(pairs):
        r = idx // 3 + 1
        c = idx % 3 + 1
        sub = joined[["date", ours_col, kf_col]].dropna().sort_values("date")
        if sub.empty:
            continue
        cum_ours = (1.0 + sub[ours_col]).cumprod() - 1.0
        cum_kf = (1.0 + sub[kf_col]).cumprod() - 1.0
        fig.add_trace(
            go.Scatter(
                x=sub["date"], y=cum_ours, mode="lines", name="Ours",
                line=dict(width=2, color="#1f77b4"), showlegend=(idx == 0),
            ),
            row=r, col=c,
        )
        fig.add_trace(
            go.Scatter(
                x=sub["date"], y=cum_kf, mode="lines", name="KF",
                line=dict(width=2, color="#ff7f0e", dash="dash"), showlegend=(idx == 0),
            ),
            row=r, col=c,
        )
    fig.update_layout(
        title="Arm 2: Cumulative factor returns --- Ours vs Ken French (2017-2024)",
        template="plotly_white",
        height=700,
        margin=dict(l=60, r=40, t=80, b=40),
    )
    for i in range(1, 7):
        fig.update_yaxes(tickformat=".1%", row=(i - 1) // 3 + 1, col=(i - 1) % 3 + 1)
    return fig


def arm2_correlation_summary_figure(diag: pd.DataFrame) -> go.Figure:
    """Horizontal-bar of per-factor Pearson correlation with 0.85 threshold."""
    diag = diag.sort_values("pearson_rho", ascending=True)
    colors = ["#2ca02c" if r > 0.85 else "#d62728" for r in diag["pearson_rho"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=diag["pearson_rho"],
            y=diag["factor"].str.upper(),
            orientation="h",
            marker=dict(color=colors),
            text=[f"{r:.3f}" for r in diag["pearson_rho"]],
            textposition="auto",
        )
    )
    fig.add_vline(x=0.85, line=dict(color="black", dash="dot"), annotation_text="0.85 threshold")
    fig.update_layout(
        title="Arm 2: Per-factor Pearson correlation with Ken French",
        xaxis_title="Pearson correlation",
        yaxis_title="Factor",
        xaxis=dict(range=[-0.2, 1.05]),
        template="plotly_white",
        height=400,
        margin=dict(l=80, r=40, t=80, b=40),
    )
    return fig
