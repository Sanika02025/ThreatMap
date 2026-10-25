"""
Hybrid retrieval engine combining BM25 lexical search and semantic vector similarity.
Supports Reciprocal Rank Fusion (RRF) and Weighted Score Fusion.
"""

from typing import Dict, List, Optional
from threatmap.logging_config import setup_logger
from threatmap.schemas import RetrievedCandidate
from threatmap.retrieval.lexical import BM25Retriever
from threatmap.retrieval.semantic import SemanticRetriever

logger = setup_logger("threatmap.hybrid")


def reciprocal_rank_fusion(
    lexical_candidates: List[RetrievedCandidate],
    semantic_candidates: List[RetrievedCandidate],
    k: int = 60,
    top_k: int = 5,
) -> List[RetrievedCandidate]:
    """
    Combines ranked lists using standard Reciprocal Rank Fusion (RRF):
    RRF_score(d) = sum_{m in systems} (1 / (k + rank_m(d)))
    """
    fused_scores: Dict[str, float] = {}
    candidate_metadata: Dict[str, Dict] = {}

    # Process lexical candidates
    for cand in lexical_candidates:
        tid = cand.technique_id
        rank = cand.lexical_rank or 999
        rrf_contrib = 1.0 / (k + rank)
        fused_scores[tid] = fused_scores.get(tid, 0.0) + rrf_contrib

        if tid not in candidate_metadata:
            candidate_metadata[tid] = {
                "name": cand.technique_name,
                "tactics": cand.tactics,
                "description": cand.description,
                "lex_rank": cand.lexical_rank,
                "lex_score": cand.lexical_score,
                "sem_rank": None,
                "sem_score": None,
            }

    # Process semantic candidates
    for cand in semantic_candidates:
        tid = cand.technique_id
        rank = cand.semantic_rank or 999
        rrf_contrib = 1.0 / (k + rank)
        fused_scores[tid] = fused_scores.get(tid, 0.0) + rrf_contrib

        if tid in candidate_metadata:
            candidate_metadata[tid]["sem_rank"] = cand.semantic_rank
            candidate_metadata[tid]["sem_score"] = cand.semantic_score
        else:
            candidate_metadata[tid] = {
                "name": cand.technique_name,
                "tactics": cand.tactics,
                "description": cand.description,
                "lex_rank": None,
                "lex_score": None,
                "sem_rank": cand.semantic_rank,
                "sem_score": cand.semantic_score,
            }

    # Sort descending by fused RRF score
    sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    fused_results: List[RetrievedCandidate] = []
    for rank, (tid, score) in enumerate(sorted_items, start=1):
        meta = candidate_metadata[tid]
        fused_results.append(
            RetrievedCandidate(
                technique_id=tid,
                technique_name=meta["name"],
                tactics=meta["tactics"],
                description=meta["description"],
                lexical_rank=meta["lex_rank"],
                lexical_score=meta["lex_score"],
                semantic_rank=meta["sem_rank"],
                semantic_score=meta["sem_score"],
                fused_rank=rank,
                fused_score=round(score, 6),
                retrieval_method="hybrid_rrf",
            )
        )

    return fused_results


def weighted_score_fusion(
    lexical_candidates: List[RetrievedCandidate],
    semantic_candidates: List[RetrievedCandidate],
    lexical_weight: float = 0.5,
    semantic_weight: float = 0.5,
    top_k: int = 5,
) -> List[RetrievedCandidate]:
    """
    Min-max normalizes lexical BM25 scores and combines them with semantic cosine similarities:
    Combined_score = w_lex * Norm(BM25) + w_sem * CosineSim
    """
    # Normalize BM25 scores to [0, 1]
    lex_scores = [c.lexical_score for c in lexical_candidates if c.lexical_score is not None]
    min_lex = min(lex_scores) if lex_scores else 0.0
    max_lex = max(lex_scores) if lex_scores else 1.0
    lex_range = (max_lex - min_lex) if (max_lex - min_lex) > 1e-6 else 1.0

    scores_map: Dict[str, float] = {}
    candidate_metadata: Dict[str, Dict] = {}

    for cand in lexical_candidates:
        tid = cand.technique_id
        raw_score = cand.lexical_score or 0.0
        norm_score = max(0.0, (raw_score - min_lex) / lex_range)
        scores_map[tid] = scores_map.get(tid, 0.0) + (lexical_weight * norm_score)

        if tid not in candidate_metadata:
            candidate_metadata[tid] = {
                "name": cand.technique_name,
                "tactics": cand.tactics,
                "description": cand.description,
                "lex_rank": cand.lexical_rank,
                "lex_score": cand.lexical_score,
                "sem_rank": None,
                "sem_score": None,
            }

    for cand in semantic_candidates:
        tid = cand.technique_id
        # Cosine similarity in [-1, 1], clip to [0, 1]
        sem_score = max(0.0, cand.semantic_score or 0.0)
        scores_map[tid] = scores_map.get(tid, 0.0) + (semantic_weight * sem_score)

        if tid in candidate_metadata:
            candidate_metadata[tid]["sem_rank"] = cand.semantic_rank
            candidate_metadata[tid]["sem_score"] = cand.semantic_score
        else:
            candidate_metadata[tid] = {
                "name": cand.technique_name,
                "tactics": cand.tactics,
                "description": cand.description,
                "lex_rank": None,
                "lex_score": None,
                "sem_rank": cand.semantic_rank,
                "sem_score": cand.semantic_score,
            }

    sorted_items = sorted(scores_map.items(), key=lambda x: x[1], reverse=True)[:top_k]

    fused_results: List[RetrievedCandidate] = []
    for rank, (tid, score) in enumerate(sorted_items, start=1):
        meta = candidate_metadata[tid]
        fused_results.append(
            RetrievedCandidate(
                technique_id=tid,
                technique_name=meta["name"],
                tactics=meta["tactics"],
                description=meta["description"],
                lexical_rank=meta["lex_rank"],
                lexical_score=meta["lex_score"],
                semantic_rank=meta["sem_rank"],
                semantic_score=meta["sem_score"],
                fused_rank=rank,
                fused_score=round(score, 4),
                retrieval_method="hybrid_weighted",
            )
        )

    return fused_results


class HybridRetriever:
    """Unified retriever exposing lexical, semantic, and hybrid search methods."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        semantic_retriever: SemanticRetriever,
        default_mode: str = "hybrid_rrf",
        rrf_k: int = 60,
    ):
        self.bm25_retriever = bm25_retriever
        self.semantic_retriever = semantic_retriever
        self.default_mode = default_mode
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        mode: Optional[str] = None,
        top_k: int = 5,
        pool_size: int = 15,
    ) -> List[RetrievedCandidate]:
        """
        Executes retrieval according to selected mode:
        'lexical', 'semantic', 'hybrid_rrf', or 'hybrid_weighted'.
        """
        retrieval_mode = mode or self.default_mode

        if retrieval_mode == "lexical":
            return self.bm25_retriever.retrieve(query, top_k=top_k)

        if retrieval_mode == "semantic":
            return self.semantic_retriever.retrieve(query, top_k=top_k)

        # For hybrid methods, retrieve larger candidate pools first
        effective_pool = max(top_k * 3, pool_size)
        lex_cands = self.bm25_retriever.retrieve(query, top_k=effective_pool)
        sem_cands = self.semantic_retriever.retrieve(query, top_k=effective_pool)

        if retrieval_mode == "hybrid_weighted":
            return weighted_score_fusion(
                lex_cands, sem_cands, lexical_weight=0.5, semantic_weight=0.5, top_k=top_k
            )

        # Default to RRF
        return reciprocal_rank_fusion(lex_cands, sem_cands, k=self.rrf_k, top_k=top_k)
