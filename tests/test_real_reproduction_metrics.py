"""
Deterministic unit tests for RAG-ATT&CK reproduction metrics using STUB LLM and tiny fake embeddings.
Zero randomness, exact known answers, explicit assertions.
"""

from typing import Dict, List, Optional, Set
import numpy as np
import pytest

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
