"""
Robust parser for LLM responses. Strips markdown fences and validates JSON schemas.
"""

import json
import re
from typing import Any, Dict, Optional
from threatmap.logging_config import setup_logger

logger = setup_logger("threatmap.prediction_parser")


def extract_json_from_text(raw_text: str) -> Optional[str]:
    """Extracts JSON substring from text, removing markdown code fences if present."""
    if not raw_text:
        return None

    cleaned = raw_text.strip()

    # Pattern for ```json ... ``` or ``` ... ```
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if code_block_match:
        return code_block_match.group(1).strip()

    # Find the outermost curly braces
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return cleaned[start_idx : end_idx + 1].strip()

    return cleaned


def parse_llm_response(raw_text: Optional[str], default_sentence_id: str = "S001") -> Dict[str, Any]:
    """
    Parses LLM output text into a structured dictionary.
    Falls back gracefully to an abstained response if parsing fails.
    """
    if not raw_text:
        return {
            "sentence_id": default_sentence_id,
            "predictions": [],
            "abstained": True,
            "abstention_reason": "No response received from local LLM.",
        }

    json_str = extract_json_from_text(raw_text)
    if not json_str:
        return {
            "sentence_id": default_sentence_id,
            "predictions": [],
            "abstained": True,
            "abstention_reason": "Could not locate JSON in LLM response.",
        }

    try:
        parsed = json.loads(json_str)
        if not isinstance(parsed, dict):
            raise ValueError("Parsed JSON root is not an object.")

        # Ensure required keys exist
        parsed["sentence_id"] = str(parsed.get("sentence_id") or default_sentence_id)
        raw_preds = parsed.get("predictions", [])
        if not isinstance(raw_preds, list):
            raw_preds = []

        clean_preds = []
        for p in raw_preds:
            if isinstance(p, dict) and "technique_id" in p:
                clean_preds.append(
                    {
                        "technique_id": str(p.get("technique_id", "")).strip(),
                        "technique_name": str(p.get("technique_name", "")).strip(),
                        "evidence_quote": str(p.get("evidence_quote", "")).strip(),
                        "explanation": str(p.get("explanation", "")).strip(),
                        "evidence_type": str(p.get("evidence_type", "explicit")),
                        "llm_confidence": (
                            float(p["llm_confidence"])
                            if p.get("llm_confidence") is not None
                            else None
                        ),
                    }
                )

        parsed["predictions"] = clean_preds
        parsed["abstained"] = bool(parsed.get("abstained", len(clean_preds) == 0))
        parsed["abstention_reason"] = parsed.get("abstention_reason")
        return parsed

    except Exception as e:
        logger.warning(f"Error parsing LLM JSON: {e}")
        return {
            "sentence_id": default_sentence_id,
            "predictions": [],
            "abstained": True,
            "abstention_reason": f"Malformed LLM JSON output: {str(e)}",
        }
