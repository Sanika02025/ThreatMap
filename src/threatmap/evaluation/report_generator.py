"""
Generates structured Markdown and JSON audit reports for ThreatMap evaluations.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
from threatmap.schemas import EvaluationMetrics


def generate_evaluation_markdown_report(
    metrics: EvaluationMetrics,
    error_analysis: Dict[str, Any],
    metadata: Dict[str, Any],
    file_path: Optional[str | Path] = None,
) -> str:
    """
    Generates a formal research-grade evaluation summary in Markdown.
    """
    lines = [
        "# ThreatMap Quantitative Evaluation & Audit Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Benchmark Dataset:** {metadata.get('dataset_name', 'TRAM Benchmark')} ({metrics.total_examples} examples)  ",
        f"**Retrieval Method:** {metadata.get('retrieval_mode', 'hybrid_rrf')} (top_k={metadata.get('top_k', 5)})  ",
        f"**Classification Mode:** {'Local LLM Validated' if metadata.get('use_llm') else 'Retrieval Candidate Fallback'}  ",
        "",
        "---",
        "",
        "## 1. Global Multilabel Classification Metrics",
        "",
        "| Metric | Score | Explanation |",
        "| :--- | :---: | :--- |",
        f"| **Micro Precision** | `{metrics.micro_precision:.4f}` | Global true positives divided by all predicted instances. |",
        f"| **Micro Recall** | `{metrics.micro_recall:.4f}` | Global true positives divided by all ground truth labels. |",
        f"| **Micro F1** | `{metrics.micro_f1:.4f}` | Harmonic mean of global precision and recall. |",
        f"| **Macro Precision** | `{metrics.macro_precision:.4f}` | Unweighted mean precision across all evaluated technique classes. |",
        f"| **Macro Recall** | `{metrics.macro_recall:.4f}` | Unweighted mean recall across all evaluated technique classes. |",
        f"| **Macro F1** | `{metrics.macro_f1:.4f}` | Unweighted harmonic mean, sensitive to rare class performance. |",
        "",
        "## 2. Information Retrieval Performance",
        "",
        "| Retrieval Metric | Score | Research Interpretation |",
        "| :--- | :---: | :--- |",
        f"| **Recall@1** | `{metrics.recall_at_1:.4f}` | Accuracy of the top-ranked technique candidate. |",
        f"| **Recall@3** | `{metrics.recall_at_3:.4f}` | Fraction of queries where ground truth is in top 3. |",
        f"| **Recall@5** | `{metrics.recall_at_5:.4f}` | Fraction of queries where ground truth is in top 5. |",
        f"| **Recall@10** | `{metrics.recall_at_10:.4f}` | Fraction of queries where ground truth is in top 10. |",
        f"| **Mean Reciprocal Rank (MRR)** | `{metrics.mrr:.4f}` | Average reciprocal rank of the first relevant technique. |",
        "",
        "## 3. Abstention & False-Positive Control",
        "",
        f"- **Coverage:** `{metrics.coverage * 100:.1f}%` ({metrics.total_examples - int(metrics.abstention_rate * metrics.total_examples)}/{metrics.total_examples} sentences predicted)",
        f"- **Abstention Rate:** `{metrics.abstention_rate * 100:.1f}%` (safely abstained on ambiguous or non-adversarial sentences)",
        f"- **Total Latency:** `{metrics.runtime_seconds:.3f} s` ({metrics.runtime_seconds / max(1, metrics.total_examples) * 1000:.1f} ms/sentence)",
        "",
        "## 4. Error Analysis Summary",
        "",
        f"- **False Positives:** {error_analysis.get('total_false_positives', 0)}",
        f"- **False Negatives:** {error_analysis.get('total_false_negatives', 0)}",
        f"- **Total Abstentions:** {error_analysis.get('total_abstentions', 0)}",
        "",
    ]

    # Sample False Positives
    fps = error_analysis.get("false_positives", [])
    if fps:
        lines.append("### Sample False Positives (Over-predictions):")
        for fp in fps[:5]:
            lines.append(
                f"- **Text**: \"{fp['text']}\"\n  - Predicted: `{fp['predicted_technique_id']}` ({fp['predicted_technique_name']})\n  - Ground Truth: `{fp['ground_truth']}`"
            )
        lines.append("")

    # Sample False Negatives
    fns = error_analysis.get("false_negatives", [])
    if fns:
        lines.append("### Sample False Negatives (Missed Detections):")
        for fn in fns[:5]:
            ret_info = f"Rank {fn['retrieved_rank']}" if fn["retrieved_in_candidates"] else "Not in top candidates"
            lines.append(
                f"- **Text**: \"{fn['text']}\"\n  - Missed Technique: `{fn['missed_technique_id']}`\n  - Retrieval Status: {ret_info}"
            )
        lines.append("")

    report_content = "\n".join(lines)

    if file_path:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(report_content)

    return report_content
