"""
Abstention and false-positive control engine.
Enforces confidence boundaries, evidence presence, and cyber-relevance filters.
"""

import re
from typing import List, Optional, Tuple
from threatmap.config import AbstentionConfig
from threatmap.schemas import RetrievedCandidate, TechniquePrediction

# Patterns identifying general non-adversarial sentences (mitigations, admin notes)
NON_ADVERSARIAL_PATTERNS = [
    r"\b(?:organizations?|administrators?|users?)\s+(?:should|must|are advised to|recommend)\b",
    r"\b(?:incident response team|forensic team)\s+(?:isolated|remediated|recovered|restored)\b",
    r"\b(?:for more information|disclaimer|copyright|published on|q[1-4]\s+20\d{2})\b",
]
COMPILED_NON_ADVERSARIAL = re.compile("|".join(NON_ADVERSARIAL_PATTERNS), re.IGNORECASE)


class AbstentionEngine:
    """Controls false positives by abstaining when evidence is insufficient or irrelevant."""

    def __init__(self, config: AbstentionConfig):
        self.config = config

    def check_retrieval_thresholds(
        self,
        candidates: List[RetrievedCandidate],
        mode: str = "hybrid_rrf",
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluates whether retrieved candidates meet minimum confidence thresholds.
        Returns: (should_abstain, reason)
        """
        if not candidates:
            return True, "No candidate techniques were retrieved."

        top_candidate = candidates[0]

        if mode == "semantic":
            if (
                top_candidate.semantic_score is not None
                and top_candidate.semantic_score < self.config.min_semantic_similarity
            ):
                return (
                    True,
                    f"Top semantic candidate ({top_candidate.technique_id}) score {top_candidate.semantic_score:.3f} below threshold {self.config.min_semantic_similarity:.3f}.",
                )

        elif mode == "hybrid_rrf":
            if top_candidate.fused_score < self.config.min_rrf_score:
                return (
                    True,
                    f"Top hybrid RRF score {top_candidate.fused_score:.4f} below threshold {self.config.min_rrf_score:.4f}.",
                )

        return False, None

    def should_abstain(
        self,
        sentence_text: str,
        candidates: List[RetrievedCandidate],
        validated_predictions: List[TechniquePrediction],
        rejection_reasons: Optional[List[str]] = None,
        retrieval_mode: str = "hybrid_rrf",
    ) -> Tuple[bool, Optional[str]]:
        """
        Comprehensive decision function for abstention.
        """
        # 1. Non-adversarial check (mitigations, administrative statements)
        if COMPILED_NON_ADVERSARIAL.search(sentence_text):
            # If the sentence explicitly talks about mitigations / advice and has weak or no predictions
            if len(validated_predictions) == 0:
                return True, "Sentence describes defensive mitigation or general advisory context."

        # 2. Check candidate retrieval thresholds
        retrieval_abstained, ret_reason = self.check_retrieval_thresholds(candidates, mode=retrieval_mode)
        if retrieval_abstained:
            return True, ret_reason

        # 3. If predictions are empty
        if not validated_predictions:
            if rejection_reasons:
                return True, f"Predictions rejected: {'; '.join(rejection_reasons)}"
            return True, "No ATT&CK technique met the evidence confidence requirements."

        return False, None
