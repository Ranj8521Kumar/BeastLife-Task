"""
AI Customer Intelligence System — Phase 7: Dashboard

Reads `data/predictions.csv` produced by `model/classifier.py` and generates:
- `outputs/dashboard_metrics.json` (submission-friendly metrics)
- PNG charts (best-effort; requires matplotlib)
- `outputs/dashboard.html` (single page summary)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


def _safe_to_dict(series: pd.Series) -> Dict[str, Any]:
    # Ensure JSON-serializable values
    return {str(k): (float(v) if pd.notna(v) else 0.0) for k, v in series.to_dict().items()}


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [
        "true_category",
        "predicted_category",
        "confidence_score",
        "confidence_tier",
        "action",
        "channel",
        "urgent_override",
        "correct",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    total = int(len(df))
    category_counts = df["true_category"].value_counts().sort_values(ascending=False)
    category_percent = ((category_counts / total) * 100.0).round(2)
    top_common_problems = [
        {"issue_type": str(cat), "query_count": int(category_counts[cat]), "query_percent": float(category_percent[cat])}
        for cat in category_counts.index[:5]
    ]

    # No timestamp column exists in predictions.csv, so use query order as a timeline proxy.
    ordered_df = df.reset_index(drop=True).copy()
    ordered_df["event_order"] = range(1, len(ordered_df) + 1)
    ordered_df["week_bucket"] = ((ordered_df["event_order"] - 1) // 7) + 1
    ordered_df["month_bucket"] = ((ordered_df["event_order"] - 1) // 30) + 1
    weekly_trend = ordered_df.groupby("week_bucket").size()
    monthly_trend = ordered_df.groupby("month_bucket").size()

    return {
        "total_queries": total,
        "category_percentage_distribution": _safe_to_dict(category_percent),
        "most_common_customer_problems": top_common_problems,
        "weekly_query_trend_proxy": {f"Week-{int(k)}": int(v) for k, v in weekly_trend.to_dict().items()},
        "monthly_query_trend_proxy": {f"Month-{int(k)}": int(v) for k, v in monthly_trend.to_dict().items()},
    }


def _inline_bar_chart(title: str, data: Dict[str, float]) -> str:
    if not data:
        return f'<div class="chart"><h3>{title}</h3><p><em>No data available.</em></p></div>'
    max_val = max(float(v) for v in data.values()) or 1.0
    rows = "\n".join(
        f"<div class='bar-row'><div class='bar-label'>{k}</div><div class='bar-wrap'><div class='bar-fill' style='width:{(float(v)/max_val)*100:.2f}%'></div></div><div class='bar-value'>{float(v):.2f}</div></div>"
        for k, v in data.items()
    )
    return f"<div class='chart'><h3>{title}</h3><div class='bar-chart'>{rows}</div></div>"


def _inline_line_chart(title: str, data: Dict[str, int], width: int = 900, height: int = 280) -> str:
    if not data:
        return f'<div class="chart"><h3>{title}</h3><p><em>No data available.</em></p></div>'
    items = list(data.items())
    values = [int(v) for _, v in items]
    max_v = max(values) if values else 1
    max_v = max(max_v, 1)
    left_pad, right_pad, top_pad, bottom_pad = 45, 20, 20, 40
    usable_w = width - left_pad - right_pad
    usable_h = height - top_pad - bottom_pad
    points = []
    for i, (_, val) in enumerate(items):
        x = left_pad + (i * usable_w / max(1, len(items) - 1))
        y = top_pad + (1 - (val / max_v)) * usable_h
        points.append((x, y, val))
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in points)
    dots = "\n".join(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='4'></circle>" for x, y, _ in points)
    labels = "\n".join(
        f"<text x='{x:.2f}' y='{height-12}' text-anchor='middle' class='axis-label'>{name}</text>"
        for (name, _), (x, _, _) in zip(items, points)
    )
    values_text = "\n".join(
        f"<text x='{x:.2f}' y='{y-8:.2f}' text-anchor='middle' class='point-value'>{val}</text>"
        for (_, _), (x, y, val) in zip(items, points)
    )
    return f"""
<div class='chart'>
  <h3>{title}</h3>
  <svg viewBox='0 0 {width} {height}' class='line-chart' role='img' aria-label='{title}'>
    <line x1='{left_pad}' y1='{height-bottom_pad}' x2='{width-right_pad}' y2='{height-bottom_pad}' class='axis'></line>
    <line x1='{left_pad}' y1='{top_pad}' x2='{left_pad}' y2='{height-bottom_pad}' class='axis'></line>
    <polyline points='{polyline}' class='trend-line'></polyline>
    {dots}
    {labels}
    {values_text}
  </svg>
</div>
"""


def render_dashboard_html(metrics: Dict[str, Any], out_html: Path) -> None:

    html_sections = []

    # Only requested outputs
    issue_pct = metrics.get("category_percentage_distribution", {})
    issue_rows = "\n".join(
        f"<tr><td>{issue}</td><td style='text-align:right'>{pct:.2f}%</td></tr>"
        for issue, pct in issue_pct.items()
    )
    common_rows = "\n".join(
        f"<tr><td>{r['issue_type']}</td><td style='text-align:right'>{r['query_count']}</td><td style='text-align:right'>{r['query_percent']:.2f}%</td></tr>"
        for r in metrics.get("most_common_customer_problems", [])
    )
    html_sections.append(
        f"""
        <section>
          <h2>Problem Distribution Dashboard</h2>
          <p class="meta"><strong>Total Queries:</strong> {metrics.get("total_queries", 0)}</p>
          <div class="grid-2">
            <div class="panel">
              <h3>% of Total Queries by Category</h3>
              <table class="data-table">
                <thead><tr><th>Issue Type</th><th class="num-col">% Queries</th></tr></thead>
                <tbody>{issue_rows}</tbody>
              </table>
            </div>
            <div class="panel">
              <h3>Most Common Customer Problems</h3>
              <table class="data-table">
                <thead><tr><th>Issue Type</th><th class="num-col">Count</th><th class="num-col">%</th></tr></thead>
                <tbody>{common_rows}</tbody>
              </table>
            </div>
          </div>
        </section>
        """
    )

    html_sections.append(
        "<section>"
        + _inline_bar_chart("% of Queries by Category", metrics.get("category_percentage_distribution", {}))
        + "</section>"
    )

    html_sections.append(
        "<section>"
        + _inline_line_chart("Weekly Trend (proxy)", metrics.get("weekly_query_trend_proxy", {}))
        + _inline_line_chart("Monthly Trend (proxy)", metrics.get("monthly_query_trend_proxy", {}))
        + "</section>"
    )


    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>AI Customer Intelligence Dashboard</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 20px; color: #111; }}
    h1 {{ margin: 0 0 8px 0; }}
    h2 {{ margin: 0 0 8px 0; }}
    h3 {{ margin: 0 0 8px 0; font-size: 20px; }}
    .meta {{ margin: 4px 0 14px 0; }}
    section {{ margin-top: 16px; }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }}
    .panel {{ border: 1px solid #ececec; border-radius: 8px; padding: 10px; background: #fff; }}
    .chart {{ margin: 12px 0; border: 1px solid #ececec; border-radius: 8px; padding: 10px; background: #fff; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 6px; table-layout: fixed; }}
    th, td {{ border-bottom: 1px solid #eee; padding: 10px 8px; text-align: left; vertical-align: middle; }}
    th {{ font-weight: 600; }}
    thead th {{ background: #f7f7f7; }}
    .num-col {{ text-align: right; width: 110px; }}
    .data-table td:nth-child(2), .data-table td:nth-child(3) {{ text-align: right; }}
    .data-table td:first-child {{ word-break: break-word; }}
    .bar-chart {{ display: grid; gap: 8px; }}
    .bar-row {{ display: grid; grid-template-columns: 210px 1fr 70px; gap: 10px; align-items: center; }}
    .bar-label {{ font-size: 13px; }}
    .bar-wrap {{ background: #eef2ff; border-radius: 999px; height: 16px; overflow: hidden; }}
    .bar-fill {{ background: #2563eb; height: 100%; border-radius: 999px; }}
    .bar-value {{ text-align: right; font-size: 13px; }}
    .line-chart {{ width: 100%; height: auto; display: block; background: #fafafa; border: 1px solid #eee; border-radius: 6px; }}
    .axis {{ stroke: #9ca3af; stroke-width: 1; }}
    .trend-line {{ fill: none; stroke: #2563eb; stroke-width: 2; }}
    .line-chart circle {{ fill: #2563eb; }}
    .axis-label {{ fill: #374151; font-size: 11px; }}
    .point-value {{ fill: #111827; font-size: 11px; }}
    @media (max-width: 900px) {{
      .grid-2 {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 1fr; }}
    }}
    code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Problem Distribution Dashboard</h1>
  {'\n'.join(html_sections)}
</body>
</html>"""

    out_html.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_csv", type=str, default=None)
    parser.add_argument("--outputs_dir", type=str, default=None)
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    predictions_csv = (
        Path(args.predictions_csv)
        if args.predictions_csv
        else root_dir / "data" / "predictions.csv"
    )
    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else root_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    df = load_predictions(predictions_csv)
    metrics = compute_metrics(df)

    # Always write metrics JSON for submission.
    metrics_path = outputs_dir / "dashboard_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # HTML dashboard
    out_html = outputs_dir / "dashboard.html"
    render_dashboard_html(metrics, out_html)

    print(f"[OK] Dashboard generated in: {outputs_dir}")
    print(f"  - {metrics_path}")
    print(f"  - {out_html}")


if __name__ == "__main__":
    main()

