"""
Validation functions for MITRE ATT&CK technique IDs and evidence grounding.
"""

import re
from typing import Optional, Set

ATTACK_ID_PATTERN = re.compile(r"^T\d{4}(\.\d{3})?$")


def is_valid_attack_id_format(technique_id: str) -> bool:
    """Checks if a string matches the standard MITRE ATT&CK ID format (e.g. T1059 or T1059.001)."""
    if not technique_id or not isinstance(technique_id, str):
        return False
    return bool(ATTACK_ID_PATTERN.match(technique_id.strip()))


def validate_technique_id(technique_id: str, valid_technique_ids: Set[str]) -> bool:
    """Validates both format and existence within the loaded knowledge base."""
    if not is_valid_attack_id_format(technique_id):
        return False
    return technique_id.strip() in valid_technique_ids


def verify_evidence_in_text(evidence_quote: str, source_text: str, min_chars: int = 10) -> bool:
    """
    Verifies that the evidence quote provided by the classifier actually exists
    within the source sentence or context, protecting against hallucinations.
    """
    if not evidence_quote or not source_text:
        return False

    clean_quote = " ".join(evidence_quote.strip().lower().split())
    clean_source = " ".join(source_text.strip().lower().split())

    if len(clean_quote) < min_chars:
        return False

    # Exact or substring containment
    if clean_quote in clean_source:
        return True

    # High token overlap check for minor punctuation differences
    quote_tokens = set(clean_quote.split())
    source_tokens = set(clean_source.split())

    if not quote_tokens:
        return False

    overlap = len(quote_tokens.intersection(source_tokens)) / len(quote_tokens)
    return overlap >= 0.85
