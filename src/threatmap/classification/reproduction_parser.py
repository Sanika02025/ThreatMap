"""
Regex-based response parser for RAG-ATT&CK (Schuurman, 2025) reproduction protocol.
Extracts 'Txxxx' and 'Txxxx.xxx' technique codes from plain text outputs,
handles DeepSeek <think> reasoning blocks, and audits unlisted technique predictions.
"""

import re
from typing import List, Optional, Set, Tuple


# Regex pattern to match technique IDs (e.g. T1059, T1059.003, T1003.001)
TECHNIQUE_ID_REGEX = re.compile(r"\b(T\d{4}(?:\.\d{3})?)\b", re.IGNORECASE)

# Regex to match 'Txxxx : Technique Name' pattern
TECHNIQUE_LINE_REGEX = re.compile(
    r"\b(T\d{4}(?:\.\d{3})?)\s*(?::|-)?\s*([^\n\r,;]+)?", re.IGNORECASE
)

# Regex to extract <think>...</think> reasoning blocks
THINK_BLOCK_REGEX = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)


def strip_think_block(text: str) -> str:
    """Removes <think>...</think> tags and enclosed reasoning text."""
    if not text:
        return ""
    return THINK_BLOCK_REGEX.sub("", text).strip()


def extract_think_block(text: str) -> str:
    """Extracts only the content within <think>...</think> tags."""
    if not text:
        return ""
    matches = THINK_BLOCK_REGEX.findall(text)
    return "\n".join(matches).strip()


class ReproductionParseResult(tuple):
    """Tuple subclass (valid, unlisted, detailed) with dict-like key access for backward compatibility."""

    def __new__(cls, valid_preds: List[str], unlisted_preds: List[str], detailed_records: List[Dict]):
        return super().__new__(cls, (valid_preds, unlisted_preds, detailed_records))

    def __getitem__(self, key):
        if isinstance(key, str):
            mapping = {
                "in_scope_predictions": self[0],
                "valid_predictions": self[0],
                "unlisted_predictions": self[1],
                "raw_predictions": self[0] + self[1],
                "detailed_records": self[2],
            }
            if key in mapping:
                return mapping[key]
            raise KeyError(key)
        return super().__getitem__(key)


def parse_reproduction_response(
    raw_response: Optional[str],
    valid_50_classes: Optional[Set[str]] = None,
    valid_classes: Optional[Set[str]] = None,
    is_deepseek_reasoning: bool = False,
    include_think_in_extraction: bool = False,
) -> ReproductionParseResult:
    """
    Parses technique predictions from raw text response.
    Returns:
        ReproductionParseResult(valid_predictions, unlisted_predictions, detailed_records)
    """
    target_classes = valid_50_classes if valid_50_classes is not None else (valid_classes or set())
    if not raw_response or raw_response.strip().lower() == "none":
        return ReproductionParseResult([], [], [])

    text_to_parse = raw_response

    # If it is DeepSeek reasoning output
    if is_deepseek_reasoning:
        if not include_think_in_extraction:
            # Paper main result: strip <think> reasoning block
            text_to_parse = strip_think_block(raw_response)
        else:
            # Paper variant: parse both <think> block and final response
            text_to_parse = raw_response

    matches = TECHNIQUE_ID_REGEX.findall(text_to_parse)

    valid_preds: List[str] = []
    unlisted_preds: List[str] = []
    detailed_records = []

    seen = set()
    for tid_raw in matches:
        tid = tid_raw.upper().strip()
        if tid in seen:
            continue
        seen.add(tid)

        if tid in target_classes:
            valid_preds.append(tid)
            detailed_records.append({"technique_id": tid, "status": "valid"})
        else:
            unlisted_preds.append(tid)
            detailed_records.append({"technique_id": tid, "status": "unlisted"})

    return ReproductionParseResult(valid_preds, unlisted_preds, detailed_records)
