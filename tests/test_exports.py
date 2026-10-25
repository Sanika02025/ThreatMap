"""
Unit tests for CSV, JSON, and JSONL exports.
"""

import json
from threatmap.export.csv_export import export_to_csv
from threatmap.export.json_export import export_to_json, export_to_jsonl
from threatmap.schemas import (
    ReportAnalysisResult,
    SentenceAnalysisResult,
    TechniquePrediction,
)


def _build_sample_report_result():
    return ReportAnalysisResult(
        document_id="DOC123",
        filename="threat_test.txt",
        timestamp="2026-10-19T09:00:00",
        total_sentences=2,
        sentences_with_predictions=1,
        abstention_count=1,
        unique_techniques=["T1059.001"],
        tactic_distribution={"Execution": 1},
        sentence_results=[
            SentenceAnalysisResult(
                sentence_id="S001",
                original_sentence="The attacker executed powershell commands.",
                predictions=[
                    TechniquePrediction(
                        technique_id="T1059.001",
                        technique_name="PowerShell",
                        tactic="Execution",
                        evidence_quote="powershell commands",
                        explanation="powershell execution",
                        retrieval_score=0.9,
                    )
                ],
                abstained=False,
            ),
            SentenceAnalysisResult(
                sentence_id="S002",
                original_sentence="Security teams recommend implementing MFA.",
                predictions=[],
                abstained=True,
                abstention_reason="Defensive recommendation",
            ),
        ],
    )


def test_export_to_csv():
    result = _build_sample_report_result()
    csv_str = export_to_csv(result)
    assert "Document ID,Sentence ID,Source Sentence" in csv_str
    assert "T1059.001,PowerShell,Execution" in csv_str
    assert "ABSTAINED" in csv_str


def test_export_to_json():
    result = _build_sample_report_result()
    json_str = export_to_json(result)
    data = json.loads(json_str)
    assert data["document_id"] == "DOC123"
    assert len(data["sentence_results"]) == 2


def test_export_to_jsonl():
    result = _build_sample_report_result()
    jsonl_str = export_to_jsonl(result)
    lines = jsonl_str.strip().split("\n")
    assert len(lines) == 2
    rec1 = json.loads(lines[0])
    assert rec1["sentence_id"] == "S001"
    rec2 = json.loads(lines[1])
    assert rec2["sentence_id"] == "S002"
    assert rec2["abstained"] is True
