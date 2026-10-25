"""
CSV exporter for ThreatMap analysis results.
"""

import csv
import io
from pathlib import Path
from typing import Optional
from threatmap.schemas import ReportAnalysisResult


def export_to_csv(result: ReportAnalysisResult, file_path: Optional[str | Path] = None) -> str:
    """
    Converts ReportAnalysisResult into a CSV string.
    Optionally saves the CSV to disk.
    """
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

    # Header
    writer.writerow(
        [
            "Document ID",
            "Sentence ID",
            "Source Sentence",
            "Technique ID",
            "Technique Name",
            "Tactic",
            "Evidence Quote",
            "Explanation",
            "Retrieval Score",
            "LLM Validated",
            "Status",
            "Abstention Reason",
            "Timestamp",
        ]
    )

    for sent_res in result.sentence_results:
        if sent_res.abstained or not sent_res.predictions:
            writer.writerow(
                [
                    result.document_id,
                    sent_res.sentence_id,
                    sent_res.original_sentence,
                    "NONE",
                    "NONE",
                    "NONE",
                    "",
                    "",
                    0.0,
                    sent_res.validated_by_llm,
                    "ABSTAINED",
                    sent_res.abstention_reason or "Low confidence / non-adversarial",
                    result.timestamp,
                ]
            )
        else:
            for pred in sent_res.predictions:
                writer.writerow(
                    [
                        result.document_id,
                        sent_res.sentence_id,
                        sent_res.original_sentence,
                        pred.technique_id,
                        pred.technique_name,
                        pred.tactic or "Unknown",
                        pred.evidence_quote,
                        pred.explanation,
                        pred.retrieval_score,
                        sent_res.validated_by_llm,
                        "PREDICTED",
                        "",
                        result.timestamp,
                    ]
                )

    csv_text = output.getvalue()

    if file_path:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(csv_text)

    return csv_text
