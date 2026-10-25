"""
Unit tests for the AbstentionEngine and false-positive control mechanisms.
"""

from threatmap.classification.abstention import AbstentionEngine
from threatmap.config import AbstentionConfig
from threatmap.schemas import RetrievedCandidate, TechniquePrediction


def test_abstention_on_non_adversarial_mitigation():
    config = AbstentionConfig(min_semantic_similarity=0.35, min_rrf_score=0.015)
    engine = AbstentionEngine(config)

    text = "Organizations are advised to enforce multi-factor authentication across remote portals."
    candidates = [
        RetrievedCandidate(
            technique_id="T1078",
            technique_name="Valid Accounts",
            tactics=["Defense Evasion"],
            description="desc",
            fused_rank=1,
            fused_score=0.02,
        )
    ]
    should_abstain, reason = engine.should_abstain(
        sentence_text=text,
        candidates=candidates,
        validated_predictions=[],  # No predictions validated
    )
    assert should_abstain is True
    assert "defensive mitigation" in reason.lower()


def test_abstention_on_low_retrieval_threshold():
    config = AbstentionConfig(min_semantic_similarity=0.40, min_rrf_score=0.02)
    engine = AbstentionEngine(config)

    candidates = [
        RetrievedCandidate(
            technique_id="T1059",
            technique_name="Command Interpreter",
            tactics=["Execution"],
            description="desc",
            semantic_score=0.22,  # Below 0.40
            fused_rank=1,
            fused_score=0.010,  # Below 0.02
        )
    ]

    should_abstain, reason = engine.check_retrieval_thresholds(candidates, mode="semantic")
    assert should_abstain is True
    assert "below threshold" in reason.lower()

    should_abstain_rrf, reason_rrf = engine.check_retrieval_thresholds(candidates, mode="hybrid_rrf")
    assert should_abstain_rrf is True
    assert "below threshold" in reason_rrf.lower()


def test_no_abstention_on_strong_prediction():
    config = AbstentionConfig(min_semantic_similarity=0.30, min_rrf_score=0.012)
    engine = AbstentionEngine(config)

    text = "Adversary launched powershell.exe with an encoded payload."
    candidates = [
        RetrievedCandidate(
            technique_id="T1059.001",
            technique_name="PowerShell",
            tactics=["Execution"],
            description="desc",
            fused_rank=1,
            fused_score=0.032,
        )
    ]
    preds = [
        TechniquePrediction(
            technique_id="T1059.001",
            technique_name="PowerShell",
            evidence_quote="powershell.exe",
            explanation="direct match",
            retrieval_score=0.032,
        )
    ]

    should_abstain, reason = engine.should_abstain(
        sentence_text=text,
        candidates=candidates,
        validated_predictions=preds,
    )
    assert should_abstain is False
    assert reason is None
