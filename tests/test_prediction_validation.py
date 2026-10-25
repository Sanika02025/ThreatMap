"""
Unit tests for LLM output parsing, rejection of hallucinated IDs, and evidence quote verification.
"""

from threatmap.classification.evidence_validator import validate_prediction_evidence
from threatmap.classification.prediction_parser import parse_llm_response
from threatmap.data.attack_loader import load_attack_data


def test_parse_llm_response_clean_json():
    raw_json = """
    {
      "sentence_id": "S001",
      "predictions": [
        {
          "technique_id": "T1059.001",
          "technique_name": "PowerShell",
          "evidence_quote": "executed powershell script",
          "explanation": "Valid PowerShell execution",
          "evidence_type": "explicit",
          "llm_confidence": 0.95
        }
      ],
      "abstained": false,
      "abstention_reason": null
    }
    """
    res = parse_llm_response(raw_json)
    assert res["abstained"] is False
    assert len(res["predictions"]) == 1
    assert res["predictions"][0]["technique_id"] == "T1059.001"
    assert res["predictions"][0]["llm_confidence"] == 0.95


def test_parse_llm_response_markdown_codeblock():
    raw_markdown = """Here is the extracted MITRE ATT&CK technique:
```json
{
  "sentence_id": "S002",
  "predictions": [],
  "abstained": true,
  "abstention_reason": "General background context"
}
```
Hope this helps!"""
    res = parse_llm_response(raw_markdown)
    assert res["abstained"] is True
    assert res["sentence_id"] == "S002"


def test_parse_llm_response_malformed():
    res = parse_llm_response("This is not JSON at all.")
    assert res["abstained"] is True
    assert "Could not locate JSON" in res["abstention_reason"] or "Malformed" in res["abstention_reason"]


def test_validate_prediction_rejects_hallucinated_id():
    kb = load_attack_data(fallback_to_fixture=True)
    raw_preds = [
        {
            "technique_id": "T9999.999",  # Invented hallucinated ID
            "technique_name": "Fake Technique",
            "evidence_quote": "attacker executed powershell",
            "explanation": "test",
        }
    ]
    source = "The attacker executed powershell to bypass policies."
    valid_preds, rejections = validate_prediction_evidence(
        raw_predictions=raw_preds,
        source_sentence=source,
        context_sentence="",
        kb=kb,
        require_valid_id=True,
    )
    assert len(valid_preds) == 0
    assert len(rejections) == 1
    assert "invalid or unknown ATT&CK ID" in rejections[0]


def test_validate_prediction_evidence_verification():
    kb = load_attack_data(fallback_to_fixture=True)
    source = "The attacker dumped credentials from the LSASS process memory."
    raw_preds = [
        {
            "technique_id": "T1003.001",
            "technique_name": "LSASS Memory",
            "evidence_quote": "dumped credentials from the LSASS process",
            "explanation": "test",
        },
        {
            "technique_id": "T1059.001",
            "technique_name": "PowerShell",
            "evidence_quote": "completely fabricated evidence not in text",
            "explanation": "test",
        },
    ]

    valid_preds, rejections = validate_prediction_evidence(
        raw_predictions=raw_preds,
        source_sentence=source,
        context_sentence="",
        kb=kb,
        verify_quote=True,
    )
    assert len(valid_preds) == 1
    assert valid_preds[0].technique_id == "T1003.001"
    assert len(rejections) == 1
    assert "Evidence quote not grounded" in rejections[0]
