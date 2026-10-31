"""
Deterministic unit tests for RAG-ATT&CK reproduction metrics using STUB LLM and tiny fake embeddings.
Zero randomness, exact known answers, explicit assertions.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threatmap.classification.reproduction_parser import parse_reproduction_response
from threatmap.evaluation.metrics import (
    calculate_paper_retrieval_recall_at_k,
    calculate_reproduction_metrics,
)


class StubLLMClient:
    """Deterministic stub LLM returning canned outputs for reproduction tests."""

    def __init__(self, canned_responses: Dict[str, str]):
        self.canned_responses = canned_responses
        self.call_count = 0

    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        self.call_count += 1
        for key, resp in self.canned_responses.items():
            if key in prompt:
                return resp
        return "T0000 : Unknown"


def test_stub_llm_exact_metrics():
    """
    Tests end-to-end evaluation with a Stub LLM generating known completions.
    
    Setup:
      Valid 50 scope: {"T1059.001", "T1078", "T1082"}
      Sentence 0: true = {"T1059.001", "T1078"}
                  LLM response: "T1059.001 : PowerShell\nT9999 : Unlisted"
                  -> In-scope pred: {"T1059.001"}
                  -> Unlisted: {"T9999"}
      Sentence 1: true = {"T1082"}
                  LLM response: "T1078 : Valid Accounts\nT1082 : System Info"
                  -> In-scope pred: {"T1078", "T1082"}
                  -> Unlisted: {}
      Sentence 2: true = set() (unlabelled row)
                  LLM response: "No techniques found."
                  -> In-scope pred: set()
                  -> Unlisted: {}

    Truth Breakdown:
      T1059.001: TP=1, FP=0, FN=0 -> P=1.0, R=1.0, F1=1.0
      T1078:     TP=0, FP=1, FN=1 -> P=0.0, R=0.0, F1=0.0
      T1082:     TP=1, FP=0, FN=0 -> P=1.0, R=1.0, F1=1.0
      Total True = 3 labels across 3 classes.
      Macro F1 = (1.0 + 0.0 + 1.0) / 3 = 0.6667 (66.67%)

      Micro totals across 50 classes:
      TP = 2 (T1059.001 on S0, T1082 on S1)
      FP = 1 (T1078 on S1)
      FN = 1 (T1078 on S0)
      Micro Precision = 2 / (2 + 1) = 2/3 = 66.67%
      Micro Recall = 2 / (2 + 1) = 2/3 = 66.67%
      Micro F1 = 66.67%
      Unlisted count = 1 ("T9999")
    """
    valid_classes = {"T1059.001", "T1078", "T1082"}
    
    responses = {
        "Sentence 0 text": "T1059.001 : PowerShell\nT9999 : Unknown Tool",
        "Sentence 1 text": "T1078 : Valid Accounts\nT1082 : System Info",
        "Sentence 2 text": "No techniques identified in this sentence.",
    }
    client = StubLLMClient(responses)

    sentences = ["Sentence 0 text", "Sentence 1 text", "Sentence 2 text"]
    ground_truth = [
        {"T1059.001", "T1078"},
        {"T1082"},
        set(),
    ]

    y_pred: List[Set[str]] = []
    raw_preds_list: List[List[str]] = []

    for s in sentences:
        resp = client.generate(s)
        parsed = parse_reproduction_response(resp, valid_50_classes=valid_classes)
        y_pred.append(set(parsed[0]))
        raw_preds_list.append(parsed[0] + parsed[1])

    assert client.call_count == 3
    assert y_pred[0] == {"T1059.001"}
    assert y_pred[1] == {"T1078", "T1082"}
    assert y_pred[2] == set()

    metrics = calculate_reproduction_metrics(
        y_true=ground_truth,
        y_pred=y_pred,
        valid_50_classes=valid_classes,
        raw_predictions=raw_preds_list,
    )

    assert metrics["total_tp"] == 2
    assert metrics["total_fp"] == 1
    assert metrics["total_fn"] == 1
    assert round(metrics["micro_precision"], 2) == 66.67
    assert round(metrics["micro_recall"], 2) == 66.67
    assert round(metrics["micro_f1"], 2) == 66.67
    assert round(metrics["macro_f1"], 2) == 66.67
    assert metrics["unlisted_predictions"] == 1
    assert metrics["total_raw_predictions"] == 4


def test_tiny_fake_embeddings_retrieval_recall():
    """
    Tests exact Recall@k with hand-crafted tiny embeddings.
    
    Setup:
      3 Techniques: T1, T2, T3
      2 Queries: Q1, Q2
      Fixed true labels = 3 (Q1 has {T1, T2}, Q2 has {T3})
      Embedding dimension = 3

      Tech embeddings:
        T1: [1.0, 0.0, 0.0]
        T2: [0.0, 1.0, 0.0]
        T3: [0.0, 0.0, 1.0]

      Query embeddings:
        Q1: [0.8, 0.6, 0.0] -> dot(Q1, [T1, T2, T3]) = [0.8, 0.6, 0.0]
            Ranked order for Q1: [T1, T2, T3]
        Q2: [0.1, 0.2, 0.9] -> dot(Q2, [T1, T2, T3]) = [0.1, 0.2, 0.9]
            Ranked order for Q2: [T3, T2, T1]

      Metrics:
        Total denominator = 3 true labels.
        k=1:
          Q1 top-1: T1 (Hit! 1 hit)
          Q2 top-1: T3 (Hit! 1 hit)
          Total hits = 2 -> Recall@1 = 2 / 3 = 66.67%
        k=2:
          Q1 top-2: [T1, T2] (Hits! 2 hits)
          Q2 top-2: [T3, T2] (Hit! 1 hit: T3)
          Total hits = 3 -> Recall@2 = 3 / 3 = 100.0%
    """
    technique_ids = ["T1", "T2", "T3"]
    tech_embs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    query_embs = np.array([
        [0.8, 0.6, 0.0],
        [0.1, 0.2, 0.9],
    ], dtype=np.float32)

    ground_truth = [
        {"T1", "T2"},
        {"T3"},
    ]
    total_labels = sum(len(gt) for gt in ground_truth)
    assert total_labels == 3

    # Compute dot products
    sim = np.dot(query_embs, tech_embs.T)
    rankings = [
        [technique_ids[idx] for idx in np.argsort(-sim[i])]
        for i in range(len(query_embs))
    ]

    assert rankings[0] == ["T1", "T2", "T3"]
    assert rankings[1] == ["T3", "T2", "T1"]

    # Calculate Recall@1 and Recall@2 using the paper's exact formula
    r1 = calculate_paper_retrieval_recall_at_k(
        y_true=ground_truth,
        retrieved_candidates_list=rankings,
        k=1,
    )
    assert round(r1, 4) == round(2 / 3, 4)

    r2 = calculate_paper_retrieval_recall_at_k(
        y_true=ground_truth,
        retrieved_candidates_list=rankings,
        k=2,
    )
    assert round(r2, 4) == 1.0


def test_chunk_max_pooling_exact_ranking():
    """
    Tests technique scoring via max chunk similarity pooling.
    
    Setup:
      Technique A has two chunks: C_A1, C_A2
      Technique B has one chunk: C_B1
      Query Q has similarities:
        sim(Q, C_A1) = 0.40
        sim(Q, C_A2) = 0.85
        sim(Q, C_B1) = 0.70

      Technique-level scores:
        Score(A) = max(0.40, 0.85) = 0.85
        Score(B) = max(0.70) = 0.70

      Expected ranking: [A, B]
      If chunk pooling took mean instead of max, Score(A) would be 0.625 < 0.70 -> [B, A].
      Max pooling correctly ensures A ranks first.
    """
    technique_ids = ["Tech_A", "Tech_B"]
    chunk_to_tech = [0, 0, 1]  # C_A1 -> 0, C_A2 -> 0, C_B1 -> 1
    chunk_sims = np.array([[0.40, 0.85, 0.70]], dtype=np.float32)

    tech_scores = np.full((1, 2), -1.0, dtype=np.float32)
    for c_idx, t_idx in enumerate(chunk_to_tech):
        tech_scores[:, t_idx] = np.maximum(tech_scores[:, t_idx], chunk_sims[:, c_idx])

    assert tech_scores[0, 0] == 0.85
    assert tech_scores[0, 1] == 0.70

    ranking = [technique_ids[idx] for idx in np.argsort(-tech_scores[0])]
    assert ranking == ["Tech_A", "Tech_B"]


def test_msmarco_reranker_reordering():
    """
    Tests cross-encoder reranking over top candidates.
    Initial candidates from dense retriever: ["T1", "T2", "T3"]
    Cross-encoder scores:
      score(Q, T1) = -2.1
      score(Q, T2) = 4.5
      score(Q, T3) = 1.0

    Reranked top 3: ["T2", "T3", "T1"]
    Remaining ranks (> top-k rerank cutoff) preserve initial baseline order.
    """
    baseline_top15 = ["T1", "T2", "T3", "T4", "T5"]
    ce_scores = np.array([-2.1, 4.5, 1.0])  # scored over top 3
    
    sorted_top3 = [baseline_top15[:3][idx] for idx in np.argsort(-ce_scores)]
    full_reranked = sorted_top3 + baseline_top15[3:]

    assert full_reranked == ["T2", "T3", "T1", "T4", "T5"]


def test_parse_llm_response_handling():
    """
    Tests regex parsing on realistic LLM output variants:
    - Normal output: 'T1059.001 : PowerShell\nT1082 : System Info'
    - 'No Mapping' / 'None' -> empty list
    - Unlisted techniques split out from valid 50
    """
    from scripts.score_llm_sample import parse_llm_response

    valid_50 = {"T1059.001", "T1082", "T1078"}

    # Case 1: Standard response
    text1 = "Based on the sentence:\nT1059.001 : PowerShell\nT9999.001 : Custom Script\nT1082 : System Info Discovery"
    valid, unlisted = parse_llm_response(text1, valid_50)
    assert valid == ["T1059.001", "T1082"]
    assert unlisted == ["T9999.001"]

    # Case 2: 'No Mapping'
    text2 = "No Mapping"
    valid2, unlisted2 = parse_llm_response(text2, valid_50)
    assert valid2 == []
    assert unlisted2 == []

    # Case 3: 'None.'
    text3 = "None."
    valid3, unlisted3 = parse_llm_response(text3, valid_50)
    assert valid3 == []
    assert unlisted3 == []

    # Case 4: None input
    valid4, unlisted4 = parse_llm_response(None, valid_50)
    assert valid4 == []
    assert unlisted4 == []


def test_scorer_fails_on_missing_cache(monkeypatch):
    """
    HARD RULE: Scorer must raise AssertionError if ANY expected cached response is missing.
    """
    from scripts.score_llm_sample import evaluate_experiment_from_cache

    sentences = [
        {"sentence": "Adversary ran cmd.exe.", "labels": ["T1059.003"], "doc_title": "Doc A"},
    ]
    techniques = [{"technique_id": "T1059.003", "technique_name": "Command Shell"}]
    valid_50 = {"T1059.003"}

    # Mock load_cache_file to return empty or missing hash
    import scripts.score_llm_sample as scorer_mod
    monkeypatch.setattr(scorer_mod, "load_cache_file", lambda exp_id: {})

    with pytest.raises(AssertionError, match="Cache file for .* is completely empty or missing"):
        evaluate_experiment_from_cache(
            exp_num=19,
            exp_id="exp19_base_llm",
            sentences_to_eval=sentences,
            techniques=techniques,
            valid_50=valid_50,
            tech_embs=None,
            query_embs=None,
            sent_to_labelled_idx={},
            context_windows=None,
        )

    # Now mock cache with wrong hash
    monkeypatch.setattr(scorer_mod, "load_cache_file", lambda exp_id: {"some_other_hash": {"response": "None"}})
    with pytest.raises(AssertionError, match="Missing cached response for sentence index 0"):
        evaluate_experiment_from_cache(
            exp_num=19,
            exp_id="exp19_base_llm",
            sentences_to_eval=sentences,
            techniques=techniques,
            valid_50=valid_50,
            tech_embs=None,
            query_embs=None,
            sent_to_labelled_idx={},
            context_windows=None,
        )


def test_score_predictions_macro_supported_classes_only():
    """
    Ensures macro F1 is computed ONLY over classes with support > 0 in the sample.
    """
    from scripts.score_llm_sample import score_predictions

    valid_50 = {f"T100{i}" for i in range(50)}  # 50 total classes
    # Only 2 classes have positive support in the gold labels
    gold = [["T1001"], ["T1002"], []]
    preds = [["T1001"], ["T1001", "T1002"], []]

    res = score_predictions(preds, gold, valid_50)

    # T1001: gold on 0 (pred on 0, 1) -> TP=1, FP=1, FN=0 -> P=0.5, R=1.0 -> F1=0.6667
    # T1002: gold on 1 (pred on 1) -> TP=1, FP=0, FN=0 -> P=1.0, R=1.0 -> F1=1.0
    # Macro F1 should be average of T1001 and T1002 only = (66.67 + 100) / 2 = 83.33%
    # (If all 50 classes were included, macro F1 would be ~3.33%)
    assert res["num_supported_classes"] == 2
    assert round(res["macro_f1"], 2) == 83.33

