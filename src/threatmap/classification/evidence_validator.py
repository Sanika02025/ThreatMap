"""
Validates prediction evidence quotes and candidate IDs against the ATT&CK knowledge base.
Ensures zero hallucinated IDs and ungrounded claims reach the final results.
"""

from typing import List, Set, Tuple
from threatmap.data.attack_loader import AttackKnowledgeBase
from threatmap.data.validators import validate_technique_id, verify_evidence_in_text
from threatmap.logging_config import setup_logger
from threatmap.schemas import TechniquePrediction

logger = setup_logger("threatmap.evidence_validator")


def validate_prediction_evidence(
    raw_predictions: List[dict],
    source_sentence: str,
    context_sentence: str,
    kb: AttackKnowledgeBase,
    require_valid_id: bool = True,
    verify_quote: bool = True,
    min_evidence_length: int = 8,
) -> Tuple[List[TechniquePrediction], List[str]]:
    """
    Validates candidate predictions against the knowledge base and verifies evidence grounding.
    Returns: (list_of_valid_predictions, list_of_rejection_reasons)
    """
    valid_ids: Set[str] = kb.get_technique_ids()
    validated_predictions: List[TechniquePrediction] = []
    rejection_reasons: List[str] = []

    combined_text = f"{context_sentence} {source_sentence}"

    for p in raw_predictions:
        tid = p.get("technique_id", "").strip()

        # 1. Validate ID existence
        if require_valid_id and not validate_technique_id(tid, valid_ids):
            msg = f"Rejected prediction with invalid or unknown ATT&CK ID: '{tid}'"
            logger.warning(msg)
            rejection_reasons.append(msg)
            continue

        tech = kb.get_technique(tid)
        tactic = tech.tactic_names[0] if (tech and tech.tactic_names) else "General"
        canonical_name = tech.technique_name if tech else p.get("technique_name", tid)

        # 2. Validate Evidence Quote
        quote = p.get("evidence_quote", "").strip()
        if verify_quote:
            is_grounded = verify_evidence_in_text(
                evidence_quote=quote,
                source_text=combined_text,
                min_chars=min_evidence_length,
            )
            if not is_grounded:
                # If quote was empty or invalid, fallback to using target sentence if relevant
                if not quote and len(source_sentence) >= min_evidence_length:
                    quote = source_sentence
                else:
                    msg = f"Rejected prediction for {tid}: Evidence quote not grounded in source text."
                    logger.warning(msg)
                    rejection_reasons.append(msg)
                    continue

        validated = TechniquePrediction(
            technique_id=tid,
            technique_name=canonical_name,
            tactic=tactic,
            evidence_quote=quote,
            explanation=p.get("explanation", "Extracted from source evidence."),
            evidence_type=p.get("evidence_type", "explicit"),
            retrieval_score=float(p.get("retrieval_score", 0.0)),
            llm_confidence=p.get("llm_confidence"),
        )
        validated_predictions.append(validated)

    return validated_predictions, rejection_reasons
