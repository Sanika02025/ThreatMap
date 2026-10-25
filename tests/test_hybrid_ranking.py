"""
Unit tests for hybrid ranking (Reciprocal Rank Fusion and Weighted Score Fusion).
"""

from threatmap.schemas import RetrievedCandidate
from threatmap.retrieval.hybrid import reciprocal_rank_fusion, weighted_score_fusion


def test_reciprocal_rank_fusion_ranking():
    lex_cands = [
        RetrievedCandidate(
            technique_id="T1059.001",
            technique_name="PowerShell",
            tactics=["Execution"],
            description="desc",
            lexical_rank=1,
            lexical_score=10.5,
            fused_rank=1,
            fused_score=10.5,
        ),
        RetrievedCandidate(
            technique_id="T1053.005",
            technique_name="Scheduled Task",
            tactics=["Persistence"],
            description="desc",
            lexical_rank=2,
            lexical_score=5.0,
            fused_rank=2,
            fused_score=5.0,
        ),
    ]

    sem_cands = [
        RetrievedCandidate(
            technique_id="T1059.001",
            technique_name="PowerShell",
            tactics=["Execution"],
            description="desc",
            semantic_rank=1,
            semantic_score=0.88,
            fused_rank=1,
            fused_score=0.88,
        ),
        RetrievedCandidate(
            technique_id="T1003.001",
            technique_name="LSASS",
            tactics=["Credential Access"],
            description="desc",
            semantic_rank=2,
            semantic_score=0.75,
            fused_rank=2,
            fused_score=0.75,
        ),
    ]

    fused = reciprocal_rank_fusion(lex_cands, sem_cands, k=60, top_k=5)
    assert len(fused) == 3
    # T1059.001 appeared at rank 1 in both lexical and semantic, so it MUST be rank 1 fused
    assert fused[0].technique_id == "T1059.001"
    assert fused[0].lexical_rank == 1
    assert fused[0].semantic_rank == 1
    # Check RRF math: 1/(60+1) + 1/(60+1) = 2/61 ~ 0.032787
    expected_rrf = round(1.0 / 61 + 1.0 / 61, 6)
    assert fused[0].fused_score == expected_rrf


def test_weighted_score_fusion():
    lex_cands = [
        RetrievedCandidate(
            technique_id="T1059.001",
            technique_name="PowerShell",
            tactics=["Execution"],
            description="desc",
            lexical_rank=1,
            lexical_score=10.0,
            fused_rank=1,
            fused_score=10.0,
        ),
        RetrievedCandidate(
            technique_id="T1053.005",
            technique_name="Scheduled Task",
            tactics=["Persistence"],
            description="desc",
            lexical_rank=2,
            lexical_score=2.0,
            fused_rank=2,
            fused_score=2.0,
        ),
    ]

    sem_cands = [
        RetrievedCandidate(
            technique_id="T1059.001",
            technique_name="PowerShell",
            tactics=["Execution"],
            description="desc",
            semantic_rank=1,
            semantic_score=0.9,
            fused_rank=1,
            fused_score=0.9,
        )
    ]

    fused = weighted_score_fusion(lex_cands, sem_cands, lexical_weight=0.5, semantic_weight=0.5, top_k=2)
    assert len(fused) == 2
    assert fused[0].technique_id == "T1059.001"
    assert fused[0].fused_rank == 1
