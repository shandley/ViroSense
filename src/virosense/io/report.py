"""Interactive HTML report generation for ViroSense results."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="{plotly_cdn}"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         color: #1a1a2e; background: #f8f9fa; line-height: 1.6; }}
  .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white; padding: 2rem; margin-bottom: 2rem; }}
  .header h1 {{ font-size: 1.8rem; margin-bottom: 0.3rem; }}
  .header p {{ opacity: 0.8; font-size: 0.95rem; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 0 1.5rem 2rem; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }}
  .grid-full {{ grid-column: 1 / -1; }}
  .card {{ background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
          padding: 1.5rem; }}
  .card h2 {{ font-size: 1.1rem; color: #16213e; margin-bottom: 1rem;
             border-bottom: 2px solid #e8ecf1; padding-bottom: 0.5rem; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
               gap: 1rem; }}
  .stat {{ text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 6px; }}
  .stat .value {{ font-size: 1.8rem; font-weight: 700; color: #1a1a2e; }}
  .stat .label {{ font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
  .plot {{ width: 100%; min-height: 350px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ background: #f1f3f5; text-align: left; padding: 0.6rem 0.8rem; cursor: pointer;
       position: sticky; top: 0; }}
  th:hover {{ background: #e2e6ea; }}
  td {{ padding: 0.5rem 0.8rem; border-bottom: 1px solid #eee; }}
  tr:hover td {{ background: #f8f9fa; }}
  .table-wrap {{ max-height: 500px; overflow-y: auto; }}
  .param-table td:first-child {{ font-weight: 600; width: 200px; color: #555; }}
  .viral {{ color: #e63946; font-weight: 600; }}
  .cellular {{ color: #457b9d; font-weight: 600; }}
  .ambiguous {{ color: #e9c46a; font-weight: 600; }}
  .search-box {{ width: 100%; padding: 0.5rem 0.8rem; border: 1px solid #dee2e6;
                border-radius: 4px; margin-bottom: 0.8rem; font-size: 0.9rem; }}
  noscript .fallback {{ padding: 2rem; text-align: center; color: #666; }}
</style>
</head>
<body>
<noscript><div class="fallback">This report requires JavaScript (Plotly) for interactive charts.
Data is also available in the companion JSON and TSV files.</div></noscript>
"""

_HTML_FOOT = "</body></html>"


def generate_detect_report(
    results: list,
    sequences: dict[str, str],
    summary: dict,
    output_path: Path,
) -> Path:
    """Generate an interactive HTML report for a detect run.

    Args:
        results: List of DetectionResult dataclasses.
        sequences: Dict of sequence_id -> DNA sequence.
        summary: Summary dict (same as detection_summary.json).
        output_path: Output directory.

    Returns:
        Path to the generated HTML file.
    """
    from dataclasses import asdict

    filepath = output_path / "detection_report.html"
    records = [asdict(r) for r in results]

    scores = [r["viral_score"] for r in records]
    lengths = [r["contig_length"] for r in records]
    classifications = [r["classification"] for r in records]
    ids = [r["contig_id"] for r in records]

    params = summary.get("parameters", {})
    clf_info = summary.get("classifier", {})
    threshold = params.get("threshold", 0.5)

    html_parts = [_HTML_HEAD.format(
        title="ViroSense Detection Report",
        plotly_cdn=PLOTLY_CDN,
    )]

    # Header
    n_viral = summary.get("n_viral", 0)
    n_cellular = summary.get("n_cellular", 0)
    n_ambiguous = summary.get("n_ambiguous", 0)
    n_total = summary.get("n_sequences", len(results))

    html_parts.append(f"""
<div class="header">
  <h1>ViroSense Detection Report</h1>
  <p>{n_total} sequences analyzed &mdash; {n_viral} viral, {n_cellular} cellular, {n_ambiguous} ambiguous</p>
</div>
<div class="container">
""")

    # Summary stats
    sd = summary.get("score_distribution", {})
    html_parts.append(f"""
<div class="grid">
  <div class="card grid-full">
    <h2>Summary</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{n_total}</div><div class="label">Total Sequences</div></div>
      <div class="stat"><div class="value viral">{n_viral}</div><div class="label">Viral</div></div>
      <div class="stat"><div class="value cellular">{n_cellular}</div><div class="label">Cellular</div></div>
      <div class="stat"><div class="value ambiguous">{n_ambiguous}</div><div class="label">Ambiguous</div></div>
      <div class="stat"><div class="value">{sd.get('above_0.9', 'N/A')}</div><div class="label">Score &gt; 0.9</div></div>
      <div class="stat"><div class="value">{sd.get('below_0.5', 'N/A')}</div><div class="label">Score &lt; 0.5</div></div>
    </div>
  </div>
""")

    # Score distribution histogram
    html_parts.append(f"""
  <div class="card">
    <h2>Score Distribution</h2>
    <div id="score-hist" class="plot"></div>
  </div>
""")

    # Classification breakdown
    html_parts.append("""
  <div class="card">
    <h2>Classification Breakdown</h2>
    <div id="class-bar" class="plot"></div>
  </div>
""")

    # Length vs score scatter
    html_parts.append("""
  <div class="card grid-full">
    <h2>Contig Length vs Viral Score</h2>
    <div id="len-score" class="plot"></div>
  </div>
""")

    # Score CDF
    html_parts.append("""
  <div class="card">
    <h2>Score Cumulative Distribution</h2>
    <div id="score-cdf" class="plot"></div>
  </div>
""")

    # Parameters
    html_parts.append(f"""
  <div class="card">
    <h2>Run Parameters</h2>
    <table class="param-table">
      <tr><td>Threshold</td><td>{threshold}</td></tr>
      <tr><td>Min Length</td><td>{params.get('min_length', 'N/A')} bp</td></tr>
      <tr><td>Backend</td><td>{params.get('backend', 'N/A')}</td></tr>
      <tr><td>Model</td><td>{params.get('model', 'N/A')}</td></tr>
      <tr><td>Layer</td><td>{params.get('layer', 'N/A')}</td></tr>
      <tr><td>Classifier Dim</td><td>{clf_info.get('input_dim', 'N/A')}-D</td></tr>
      <tr><td>Classes</td><td>{', '.join(clf_info.get('class_names', []))}</td></tr>
      <tr><td>Calibrated</td><td>{clf_info.get('calibrated', 'N/A')}</td></tr>
    </table>
  </div>
""")

    # Per-contig table
    html_parts.append("""
  <div class="card grid-full">
    <h2>Per-Contig Results</h2>
    <input type="text" class="search-box" id="table-search"
           placeholder="Search contigs..." oninput="filterTable()">
    <div class="table-wrap">
      <table id="contig-table">
        <thead><tr>
          <th onclick="sortTable(0)">Contig ID</th>
          <th onclick="sortTable(1)">Length (bp)</th>
          <th onclick="sortTable(2)">Viral Score</th>
          <th onclick="sortTable(3)">Classification</th>
        </tr></thead>
        <tbody>
""")

    for r in sorted(records, key=lambda x: -x["viral_score"]):
        cls_class = "viral" if r["classification"] not in ("cellular", "chromosome", "plasmid", "ambiguous") else r["classification"]
        if cls_class in ("chromosome", "plasmid"):
            cls_class = "cellular"
        html_parts.append(
            f'          <tr><td>{r["contig_id"]}</td><td>{r["contig_length"]}</td>'
            f'<td>{r["viral_score"]:.4f}</td>'
            f'<td class="{cls_class}">{r["classification"]}</td></tr>\n'
        )

    html_parts.append("""
        </tbody>
      </table>
    </div>
  </div>
</div>
</div>
""")

    # JavaScript for plots and table interactivity
    scores_json = json.dumps(scores)
    lengths_json = json.dumps(lengths)
    classifications_json = json.dumps(classifications)
    ids_json = json.dumps(ids)

    # Color map for classifications
    html_parts.append(f"""
<script>
const scores = {scores_json};
const lengths = {lengths_json};
const classifications = {classifications_json};
const ids = {ids_json};
const threshold = {threshold};

const colorMap = {{
  'viral': '#e63946', 'phage': '#e63946', 'rna_virus': '#c1121f',
  'cellular': '#457b9d', 'chromosome': '#457b9d', 'plasmid': '#1d3557',
  'ambiguous': '#e9c46a'
}};
const colors = classifications.map(c => colorMap[c] || '#888');

// Score histogram
Plotly.newPlot('score-hist', [{{
  x: scores, type: 'histogram', nbinsx: 50,
  marker: {{ color: '#1a1a2e', opacity: 0.8 }}
}}], {{
  xaxis: {{ title: 'Viral Score', range: [0, 1] }},
  yaxis: {{ title: 'Count' }},
  shapes: [{{ type: 'line', x0: threshold, x1: threshold, y0: 0, y1: 1,
              yref: 'paper', line: {{ color: '#e63946', width: 2, dash: 'dash' }} }}],
  annotations: [{{ x: threshold, y: 1, yref: 'paper', text: 'threshold',
                   showarrow: false, yanchor: 'bottom', font: {{ color: '#e63946', size: 11 }} }}],
  margin: {{ t: 20, r: 20 }}
}}, {{ responsive: true }});

// Classification bar
const counts = {{}};
classifications.forEach(c => counts[c] = (counts[c] || 0) + 1);
const labels = Object.keys(counts).sort();
const values = labels.map(l => counts[l]);
const barColors = labels.map(l => colorMap[l] || '#888');

Plotly.newPlot('class-bar', [{{
  x: labels, y: values, type: 'bar',
  marker: {{ color: barColors }}
}}], {{
  yaxis: {{ title: 'Count' }},
  margin: {{ t: 20, r: 20 }}
}}, {{ responsive: true }});

// Length vs score scatter
Plotly.newPlot('len-score', [{{
  x: lengths, y: scores, mode: 'markers', type: 'scatter',
  text: ids,
  marker: {{ color: colors, size: 5, opacity: 0.6 }},
  hovertemplate: '%{{text}}<br>Length: %{{x}} bp<br>Score: %{{y:.4f}}<extra></extra>'
}}], {{
  xaxis: {{ title: 'Contig Length (bp)', type: 'log' }},
  yaxis: {{ title: 'Viral Score', range: [-0.05, 1.05] }},
  shapes: [{{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: threshold, y1: threshold,
              line: {{ color: '#e63946', width: 1.5, dash: 'dash' }} }}],
  margin: {{ t: 20, r: 20 }}
}}, {{ responsive: true }});

// Score CDF
const sorted_scores = [...scores].sort((a, b) => a - b);
const cdf_y = sorted_scores.map((_, i) => (i + 1) / sorted_scores.length);
Plotly.newPlot('score-cdf', [{{
  x: sorted_scores, y: cdf_y, type: 'scatter', mode: 'lines',
  line: {{ color: '#1a1a2e', width: 2 }}
}}], {{
  xaxis: {{ title: 'Viral Score', range: [0, 1] }},
  yaxis: {{ title: 'Cumulative Fraction' }},
  shapes: [{{ type: 'line', x0: threshold, x1: threshold, y0: 0, y1: 1,
              line: {{ color: '#e63946', width: 1.5, dash: 'dash' }} }}],
  margin: {{ t: 20, r: 20 }}
}}, {{ responsive: true }});

// Table sort and filter
function sortTable(col) {{
  const table = document.getElementById('contig-table');
  const tbody = table.querySelector('tbody');
  const rows = Array.from(tbody.rows);
  const isNum = col === 1 || col === 2;
  rows.sort((a, b) => {{
    let va = a.cells[col].textContent;
    let vb = b.cells[col].textContent;
    if (isNum) return parseFloat(vb) - parseFloat(va);
    return va.localeCompare(vb);
  }});
  rows.forEach(r => tbody.appendChild(r));
}}

function filterTable() {{
  const query = document.getElementById('table-search').value.toLowerCase();
  const rows = document.querySelectorAll('#contig-table tbody tr');
  rows.forEach(r => {{
    r.style.display = r.textContent.toLowerCase().includes(query) ? '' : 'none';
  }});
}}
</script>
""")

    html_parts.append(_HTML_FOOT)

    with open(filepath, "w") as f:
        f.write("".join(html_parts))

    logger.info(f"Wrote detection report to {filepath}")
    return filepath


def generate_training_report(
    metrics: dict,
    output_path: Path,
    y_test: "np.ndarray | None" = None,
    probas_test: "np.ndarray | None" = None,
) -> Path:
    """Generate an interactive HTML report for a training/build-reference run.

    Args:
        metrics: Metrics dict from train_classifier().
        output_path: Output directory.
        y_test: Test set labels (for ROC/calibration curves).
        probas_test: Test set predicted probabilities.

    Returns:
        Path to the generated HTML file.
    """
    filepath = output_path / "training_report.html"

    n_classes = metrics.get("n_classes", 2)
    class_names = metrics.get("class_names", [])

    html_parts = [_HTML_HEAD.format(
        title="ViroSense Training Report",
        plotly_cdn=PLOTLY_CDN,
    )]

    task = metrics.get("task", "classification")
    html_parts.append(f"""
<div class="header">
  <h1>ViroSense Training Report</h1>
  <p>Task: {task} &mdash; {n_classes} classes ({', '.join(class_names)})</p>
</div>
<div class="container">
<div class="grid">
""")

    # Summary stats
    html_parts.append(f"""
  <div class="card grid-full">
    <h2>Performance Summary</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{metrics.get('accuracy', 0):.3f}</div><div class="label">Accuracy</div></div>
      <div class="stat"><div class="value">{metrics.get('f1', 0):.3f}</div><div class="label">F1 (weighted)</div></div>
      <div class="stat"><div class="value">{metrics.get('precision', 0):.3f}</div><div class="label">Precision</div></div>
      <div class="stat"><div class="value">{metrics.get('recall', 0):.3f}</div><div class="label">Recall</div></div>
      <div class="stat"><div class="value">{metrics.get('auc', 'N/A') if metrics.get('auc') is None else f"{metrics['auc']:.3f}"}</div><div class="label">AUC</div></div>
      <div class="stat"><div class="value">{metrics.get('n_train', 0)}</div><div class="label">Train</div></div>
      <div class="stat"><div class="value">{metrics.get('n_cal', 0)}</div><div class="label">Calibrate</div></div>
      <div class="stat"><div class="value">{metrics.get('n_test', 0)}</div><div class="label">Test</div></div>
    </div>
  </div>
""")

    # Confusion matrix
    cm = metrics.get("confusion_matrix")
    if cm:
        cm_json = json.dumps(cm)
        cm_labels = json.dumps(class_names or [str(i) for i in range(len(cm))])
        html_parts.append(f"""
  <div class="card">
    <h2>Confusion Matrix</h2>
    <div id="cm-heatmap" class="plot"></div>
  </div>
""")

    # Per-class metrics
    per_class = metrics.get("per_class")
    if per_class:
        pc_names = json.dumps(list(per_class.keys()))
        pc_f1 = json.dumps([v["f1"] for v in per_class.values()])
        pc_prec = json.dumps([v["precision"] for v in per_class.values()])
        pc_rec = json.dumps([v["recall"] for v in per_class.values()])
        html_parts.append(f"""
  <div class="card">
    <h2>Per-Class Metrics</h2>
    <div id="perclass-bar" class="plot"></div>
  </div>
""")

    # ROC curve (binary only, needs y_test and probas_test)
    has_roc = n_classes == 2 and y_test is not None and probas_test is not None
    if has_roc:
        import numpy as np
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_test, probas_test[:, 1])
        fpr_json = json.dumps([round(float(x), 4) for x in fpr])
        tpr_json = json.dumps([round(float(x), 4) for x in tpr])
        auc_val = metrics.get("auc", 0)
        html_parts.append(f"""
  <div class="card">
    <h2>ROC Curve (AUC = {auc_val:.3f})</h2>
    <div id="roc-curve" class="plot"></div>
  </div>
""")

    # Calibration reliability diagram (binary only)
    has_cal = n_classes == 2 and y_test is not None and probas_test is not None
    if has_cal:
        import numpy as np

        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accuracies = []
        bin_counts = []
        probs = probas_test[:, 1]
        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
            n_in = mask.sum()
            if n_in > 0:
                bin_centers.append(round(float((lo + hi) / 2), 2))
                bin_accuracies.append(round(float(y_test[mask].mean()), 4))
                bin_counts.append(int(n_in))

        cal_centers_json = json.dumps(bin_centers)
        cal_acc_json = json.dumps(bin_accuracies)
        cal_counts_json = json.dumps(bin_counts)

        brier_uncal = metrics.get("brier_score_uncalibrated", "N/A")
        brier_cal = metrics.get("brier_score", "N/A")
        ece_uncal = metrics.get("ece_uncalibrated", "N/A")
        ece_cal = metrics.get("ece", "N/A")

        html_parts.append(f"""
  <div class="card">
    <h2>Calibration Reliability Diagram</h2>
    <div id="cal-diagram" class="plot"></div>
    <table class="param-table" style="margin-top: 1rem;">
      <tr><td>Brier (uncalibrated)</td><td>{brier_uncal if isinstance(brier_uncal, str) else f'{brier_uncal:.4f}'}</td></tr>
      <tr><td>Brier (calibrated)</td><td>{brier_cal if isinstance(brier_cal, str) else f'{brier_cal:.4f}'}</td></tr>
      <tr><td>ECE (uncalibrated)</td><td>{ece_uncal if isinstance(ece_uncal, str) else f'{ece_uncal:.4f}'}</td></tr>
      <tr><td>ECE (calibrated)</td><td>{ece_cal if isinstance(ece_cal, str) else f'{ece_cal:.4f}'}</td></tr>
    </table>
  </div>
""")

    # Training parameters
    html_parts.append(f"""
  <div class="card">
    <h2>Training Parameters</h2>
    <table class="param-table">
      <tr><td>Epochs</td><td>{metrics.get('epochs', 'N/A')}</td></tr>
      <tr><td>Learning Rate</td><td>{metrics.get('lr', 'N/A')}</td></tr>
      <tr><td>Calibration</td><td>{'Yes' if metrics.get('n_cal', 0) > 0 else 'No (too few samples)'}</td></tr>
    </table>
  </div>
""")

    html_parts.append("</div></div>")

    # JavaScript for plots
    html_parts.append("<script>\n")

    if cm:
        html_parts.append(f"""
const cm = {cm_json};
const cmLabels = {cm_labels};
// Normalize for annotation text
const cmNorm = cm.map(row => {{
  const s = row.reduce((a,b) => a+b, 0);
  return row.map(v => s > 0 ? (v/s*100).toFixed(1) + '%' : '0%');
}});
const cmText = cm.map((row, i) => row.map((v, j) => v + '\\n(' + cmNorm[i][j] + ')'));

Plotly.newPlot('cm-heatmap', [{{
  z: cm, x: cmLabels, y: cmLabels,
  type: 'heatmap', colorscale: 'Blues',
  text: cmText, texttemplate: '%{{text}}', textfont: {{ size: 12 }},
  hovertemplate: 'True: %{{y}}<br>Predicted: %{{x}}<br>Count: %{{z}}<extra></extra>'
}}], {{
  xaxis: {{ title: 'Predicted', side: 'bottom' }},
  yaxis: {{ title: 'True', autorange: 'reversed' }},
  margin: {{ t: 20, r: 20 }}
}}, {{ responsive: true }});
""")

    if per_class:
        html_parts.append(f"""
const pcNames = {pc_names};
Plotly.newPlot('perclass-bar', [
  {{ x: pcNames, y: {pc_f1}, name: 'F1', type: 'bar', marker: {{ color: '#264653' }} }},
  {{ x: pcNames, y: {pc_prec}, name: 'Precision', type: 'bar', marker: {{ color: '#2a9d8f' }} }},
  {{ x: pcNames, y: {pc_rec}, name: 'Recall', type: 'bar', marker: {{ color: '#e9c46a' }} }}
], {{
  barmode: 'group',
  yaxis: {{ title: 'Score', range: [0, 1.05] }},
  margin: {{ t: 20, r: 20 }}
}}, {{ responsive: true }});
""")

    if has_roc:
        html_parts.append(f"""
Plotly.newPlot('roc-curve', [
  {{ x: {fpr_json}, y: {tpr_json}, type: 'scatter', mode: 'lines',
     name: 'ROC (AUC={auc_val:.3f})', line: {{ color: '#1a1a2e', width: 2 }} }},
  {{ x: [0, 1], y: [0, 1], type: 'scatter', mode: 'lines',
     name: 'Random', line: {{ color: '#ccc', dash: 'dash' }} }}
], {{
  xaxis: {{ title: 'False Positive Rate' }},
  yaxis: {{ title: 'True Positive Rate' }},
  margin: {{ t: 20, r: 20 }}
}}, {{ responsive: true }});
""")

    if has_cal:
        html_parts.append(f"""
Plotly.newPlot('cal-diagram', [
  {{ x: {cal_centers_json}, y: {cal_acc_json}, type: 'scatter', mode: 'lines+markers',
     name: 'Model', marker: {{ size: 8, color: '#1a1a2e' }},
     text: {cal_counts_json}, hovertemplate: 'Predicted: %{{x}}<br>Observed: %{{y:.3f}}<br>N=%{{text}}<extra></extra>' }},
  {{ x: [0, 1], y: [0, 1], type: 'scatter', mode: 'lines',
     name: 'Perfect', line: {{ color: '#ccc', dash: 'dash' }} }}
], {{
  xaxis: {{ title: 'Mean Predicted Probability' }},
  yaxis: {{ title: 'Fraction Positive', range: [0, 1.05] }},
  margin: {{ t: 20, r: 20 }}
}}, {{ responsive: true }});
""")

    html_parts.append("</script>\n")
    html_parts.append(_HTML_FOOT)

    with open(filepath, "w") as f:
        f.write("".join(html_parts))

    logger.info(f"Wrote training report to {filepath}")
    return filepath
