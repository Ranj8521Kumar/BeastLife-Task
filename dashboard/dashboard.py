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
from typing import Dict, Any, List, Tuple

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
    accuracy = float(df["correct"].mean())

    action_dist = _safe_to_dict(df["action"].value_counts())
    confidence_dist = _safe_to_dict(df["confidence_tier"].value_counts())

    urgent_override_rate = float(df["urgent_override"].mean())

    volume_by_true_category = _safe_to_dict(df["true_category"].value_counts())
    volume_by_channel = _safe_to_dict(df["channel"].value_counts())

    avg_conf_by_channel = (
        df.groupby("channel")["confidence_score"].mean().sort_values(ascending=False).round(4)
    )
    avg_conf_by_true_category = (
        df.groupby("true_category")["confidence_score"].mean().sort_values(ascending=False).round(4)
    )

    # Misclassification analysis
    mis_df = df[df["true_category"] != df["predicted_category"]].copy()
    mis_pairs = (
        mis_df.groupby(["true_category", "predicted_category"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    top_confused_pairs = mis_pairs.head(10).to_dict(orient="records")

    # Confusion matrix (counts)
    confusion = pd.crosstab(df["true_category"], df["predicted_category"])

    # SLA forecast proxy: confidence_tier -> expected priority/SLA tier
    sla_forecast = (
        df.groupby("confidence_tier").size().sort_values(ascending=False).to_dict()
    )

    return {
        "total_queries": total,
        "accuracy": round(accuracy, 4),
        "action_distribution": action_dist,
        "confidence_tier_distribution": confidence_dist,
        "urgent_override_rate": round(urgent_override_rate, 4),
        "volume_by_true_category": volume_by_true_category,
        "volume_by_channel": volume_by_channel,
        "avg_confidence_by_channel": _safe_to_dict(avg_conf_by_channel),
        "avg_confidence_by_true_category": _safe_to_dict(avg_conf_by_true_category),
        "top_confused_pairs": top_confused_pairs,
        "confusion_matrix_counts": confusion.to_dict(orient="index"),
        "sla_forecast_proxy_by_confidence_tier": {str(k): int(v) for k, v in sla_forecast.items()},
    }


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore

        return plt, np
    except Exception:
        return None, None


def plot_bar(
    x_labels: List[str],
    y_values: List[float],
    title: str,
    out_path: Path,
    plt_module,
    rotate_x: bool = True,
) -> None:
    plt = plt_module
    fig = plt.figure(figsize=(10, 5))
    plt.bar(x_labels, y_values)
    plt.title(title)
    plt.ylabel("Count")
    plt.xlabel("Category")
    if rotate_x:
        plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_horizontal_bar(
    x_labels: List[str],
    y_values: List[float],
    title: str,
    out_path: Path,
    plt_module,
) -> None:
    plt = plt_module
    fig = plt.figure(figsize=(10, 6))
    plt.barh(x_labels, y_values)
    plt.title(title)
    plt.xlabel("Average Confidence")
    plt.ylabel("Group")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(confusion: pd.DataFrame, out_path: Path, plt_module, np_module) -> None:
    plt = plt_module
    np = np_module

    fig = plt.figure(figsize=(9, 7))
    mat = confusion.values
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto", cmap="Blues")

    ax.set_title("Confusion Matrix (counts)")
    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("True Category")

    ax.set_xticks(range(len(confusion.columns)))
    ax.set_yticks(range(len(confusion.index)))
    ax.set_xticklabels(confusion.columns, rotation=35, ha="right")
    ax.set_yticklabels(confusion.index)

    # Add count labels (only if matrix is small)
    if mat.size <= 100:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, int(mat[i, j]), ha="center", va="center", color="black", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def render_dashboard_html(
    metrics: Dict[str, Any], chart_files: Dict[str, str], out_html: Path
) -> None:
    def _img_tag(filename: str, alt: str) -> str:
        if not filename:
            return f'<div class="chart"><h3>{alt}</h3><p><em>Chart not generated (matplotlib unavailable in this environment).</em></p></div>'
        return f'<div class="chart"><h3>{alt}</h3><img src="{filename}" alt="{alt}"/></div>'

    html_sections = []

    # KPI section
    kpi = metrics
    html_sections.append(
        """
        <section>
          <h2>Routing Overview</h2>
          <div class="kpi-grid">
            <div class="kpi"><div class="kpi-label">Total Queries</div><div class="kpi-value">{total}</div></div>
            <div class="kpi"><div class="kpi-label">Accuracy (correct)</div><div class="kpi-value">{acc}</div></div>
            <div class="kpi"><div class="kpi-label">Urgent Override Rate</div><div class="kpi-value">{urgent}</div></div>
          </div>
        </section>
        """.format(
            total=kpi["total_queries"],
            acc=f'{kpi["accuracy"]:.4f}',
            urgent=f'{kpi["urgent_override_rate"]:.4f}',
        )
    )

    # Charts
    html_sections.append(
        "<section>"
        + _img_tag(chart_files.get("action_dist", ""), "Action Distribution")
        + _img_tag(chart_files.get("confidence_dist", ""), "Confidence Tier Distribution")
        + "</section>"
    )

    html_sections.append(
        "<section>"
        + _img_tag(chart_files.get("volume_by_category", ""), "Volume by True Category")
        + _img_tag(chart_files.get("volume_by_channel", ""), "Volume by Channel")
        + "</section>"
    )

    html_sections.append(
        "<section>"
        + _img_tag(chart_files.get("avg_conf_channel", ""), "Avg Confidence by Channel")
        + "</section>"
    )

    html_sections.append(
        "<section>"
        + _img_tag(chart_files.get("confusion_matrix", ""), "Misclassification Heatmap")
        + "</section>"
    )

    # Top confused pairs table
    top_pairs = metrics.get("top_confused_pairs", [])
    rows = "\n".join(
        f"<tr><td>{r['true_category']}</td><td>{r['predicted_category']}</td><td style='text-align:right'>{r['count']}</td></tr>"
        for r in top_pairs
    )
    html_sections.append(
        f"""
        <section>
          <h2>Top Confused Category Pairs</h2>
          <table>
            <thead><tr><th>True</th><th>Predicted</th><th>Count</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </section>
        """
    )

    # SLA forecast proxy
    sla = metrics.get("sla_forecast_proxy_by_confidence_tier", {})
    sla_rows = "\n".join(f"<tr><td>{k}</td><td style='text-align:right'>{v}</td></tr>" for k, v in sla.items())
    html_sections.append(
        f"""
        <section>
          <h2>SLA Forecast Proxy</h2>
          <p>Proxy derived from <code>confidence_tier</code> only (not real SLA adherence).</p>
          <table>
            <thead><tr><th>Confidence Tier</th><th>Queries</th></tr></thead>
            <tbody>{sla_rows}</tbody>
          </table>
        </section>
        """
    )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>AI Customer Intelligence Dashboard</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 24px; color: #111; }}
    h1 {{ margin-bottom: 6px; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
    .kpi {{ border: 1px solid #ddd; border-radius: 10px; padding: 12px; }}
    .kpi-label {{ font-size: 12px; color: #555; margin-bottom: 4px; }}
    .kpi-value {{ font-size: 22px; font-weight: bold; }}
    section {{ margin-top: 18px; padding-top: 6px; }}
    .chart {{ margin: 12px 0; }}
    img {{ max-width: 980px; width: 100%; border: 1px solid #eee; border-radius: 10px; padding: 8px; background: #fafafa; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th, td {{ border-bottom: 1px solid #eee; padding: 10px 8px; text-align: left; }}
    thead th {{ background: #f7f7f7; }}
    code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>AI Customer Intelligence — Dashboard</h1>
  <p>Generated from <code>data/predictions.csv</code>.</p>
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

    chart_files: Dict[str, str] = {}

    plt, np = _try_import_matplotlib()
    if plt is None or np is None:
        # Matplotlib not available: only metrics JSON + HTML without images.
        chart_files = {
            "action_dist": "",
            "confidence_dist": "",
            "volume_by_category": "",
            "volume_by_channel": "",
            "avg_conf_channel": "",
            "confusion_matrix": "",
        }
        render_dashboard_html(metrics, chart_files, outputs_dir / "dashboard.html")
        print(f"[WARN] matplotlib not available. Wrote metrics + HTML only to {outputs_dir}")
        return

    # Charts
    action_dist = df["action"].value_counts().sort_values(ascending=False)
    confidence_dist = df["confidence_tier"].value_counts().sort_values(ascending=False)
    volume_by_category = df["true_category"].value_counts().sort_values(ascending=False)
    volume_by_channel = df["channel"].value_counts().sort_values(ascending=False)

    # Action distribution
    action_png = outputs_dir / "chart_action_distribution.png"
    plot_bar(
        list(action_dist.index),
        list(action_dist.values),
        "Action Distribution",
        action_png,
        plt,
    )
    chart_files["action_dist"] = action_png.name

    # Confidence distribution
    conf_png = outputs_dir / "chart_confidence_tier_distribution.png"
    plot_bar(
        list(confidence_dist.index),
        list(confidence_dist.values),
        "Confidence Tier Distribution",
        conf_png,
        plt,
    )
    chart_files["confidence_dist"] = conf_png.name

    # Volume by true category
    cat_png = outputs_dir / "chart_volume_by_true_category.png"
    plot_bar(
        list(volume_by_category.index),
        list(volume_by_category.values),
        "Volume by True Category",
        cat_png,
        plt,
    )
    chart_files["volume_by_category"] = cat_png.name

    # Volume by channel
    ch_png = outputs_dir / "chart_volume_by_channel.png"
    plot_bar(
        list(volume_by_channel.index),
        list(volume_by_channel.values),
        "Volume by Channel",
        ch_png,
        plt,
    )
    chart_files["volume_by_channel"] = ch_png.name

    # Avg confidence by channel (horizontal bar)
    avg_conf_by_channel = (
        df.groupby("channel")["confidence_score"].mean().sort_values(ascending=False).round(4)
    )
    avg_conf_png = outputs_dir / "chart_avg_confidence_by_channel.png"
    plot_horizontal_bar(
        list(avg_conf_by_channel.index),
        list(avg_conf_by_channel.values),
        "Average Confidence by Channel",
        avg_conf_png,
        plt,
    )
    chart_files["avg_conf_channel"] = avg_conf_png.name

    # Confusion matrix heatmap
    confusion = pd.crosstab(df["true_category"], df["predicted_category"])
    cm_png = outputs_dir / "chart_confusion_matrix.png"
    plot_confusion_matrix(confusion, cm_png, plt, np)
    chart_files["confusion_matrix"] = cm_png.name

    # HTML dashboard
    out_html = outputs_dir / "dashboard.html"
    render_dashboard_html(metrics, chart_files, out_html)

    print(f"[OK] Dashboard generated in: {outputs_dir}")
    print(f"  - {metrics_path}")
    print(f"  - {out_html}")


if __name__ == "__main__":
    main()

