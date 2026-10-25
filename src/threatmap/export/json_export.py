"""
JSON and JSONL exporter for ThreatMap analysis results and audit trails.
"""

import json
from pathlib import Path
from typing import Optional
from threatmap.schemas import ReportAnalysisResult


def export_to_json(
    result: ReportAnalysisResult,
    file_path: Optional[str | Path] = None,
    indent: int = 2,
) -> str:
    """Serializes ReportAnalysisResult into JSON string and optionally writes to file."""
    data = result.model_dump()
    json_text = json.dumps(data, indent=indent, ensure_ascii=False)

    if file_path:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(json_text)

    return json_text


def export_to_jsonl(
    result: ReportAnalysisResult,
    file_path: Optional[str | Path] = None,
) -> str:
    """Serializes per-sentence analysis results into line-delimited JSON (JSONL)."""
    lines = []
    for sent_res in result.sentence_results:
        record = {
            "document_id": result.document_id,
            "timestamp": result.timestamp,
            **sent_res.model_dump(),
        }
        lines.append(json.dumps(record, ensure_ascii=False))

    jsonl_text = "\n".join(lines)

    if file_path:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(jsonl_text)

    return jsonl_text
