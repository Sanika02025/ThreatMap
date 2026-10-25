"""
Optional Cross-Encoder reranker for secondary candidate ranking.
Gracefully falls back to existing candidate ranking if unavailable or disabled.
"""

from typing import List, Optional
from threatmap.logging_config import setup_logger
from threatmap.schemas import RetrievedCandidate

logger = setup_logger("threatmap.reranker")


class CrossEncoderReranker:
    """Reranks retrieved ATT&CK candidates using a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enabled: bool = False,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.enabled = enabled
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy load CrossEncoder model."""
        if self._model is None and self.enabled:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading CrossEncoder reranker: {self.model_name}")
                self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception as e:
                logger.warning(f"Could not load CrossEncoder model ({e}). Disabling reranker.")
                self.enabled = False
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedCandidate],
        top_k: Optional[int] = None,
    ) -> List[RetrievedCandidate]:
        """
        Reranks a list of candidate techniques against the query sentence.
        If disabled or missing, returns candidates unchanged.
        """
        if not self.enabled or not candidates or self.model is None:
            return candidates[:top_k] if top_k else candidates

        pairs = [
            (query, f"{c.technique_name}: {c.description[:300]}")
            for c in candidates
        ]

        try:
            scores = self.model.predict(pairs)
            for cand, score in zip(candidates, scores):
                cand.rerank_score = round(float(score), 4)

            # Sort descending by reranker score
            reranked = sorted(candidates, key=lambda c: c.rerank_score or -999.0, reverse=True)
            for idx, c in enumerate(reranked, start=1):
                c.fused_rank = idx

            return reranked[:top_k] if top_k else reranked

        except Exception as e:
            logger.error(f"Error during reranking: {e}. Falling back to original ranking.")
            return candidates[:top_k] if top_k else candidates
