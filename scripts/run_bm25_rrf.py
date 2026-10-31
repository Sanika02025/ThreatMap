"""
run_bm25_rrf.py — EXTENSION: BM25 + RRF Fusion over 50 MITRE ATT&CK techniques.

Label: EXTENSION — not in paper (Schuurman 2025).

Evaluates:
  1. BM25 alone (rank-bm25) over 50 technique descriptions vs 4,070 labelled sentences
  2. Nomic alone (with prefixes, primary setting) — from pre-computed embeddings
  3. BM25 + Nomic RRF (k=60) fusion

Reports Recall@1/5/10/20 for each, next to paper's nomic baseline (R@10=80.28).
Saves results to artifacts/bm25_rrf_results.json.

Hard rules:
  - No np.random / random / hardcoded results.
  - BM25 and embeddings computed from real data.
  - Nomic embeddings loaded from pre-computed artifacts/embedding_cache/ (no re-computation).
  - Denominator = 5,143 (total label instances across all 4,070 sentences in 50-class scope).
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank-bm25 is required: pip install rank-bm25")

BASE_DIR = Path(__file__).resolve().parent.parent

KB_PATH = BASE_DIR / "data" / "processed" / "mitre_attack_50_techniques.json"
TRAM_PATH = BASE_DIR / "data" / "tram2-data" / "multi_label.json"
EMB_CACHE_DIR = BASE_DIR / "artifacts" / "embedding_cache"
OUT_FILE = BASE_DIR / "artifacts" / "bm25_rrf_results.json"

# Paper baseline for comparison
PAPER_NOMIC_R10 = 80.28  # nomic full descriptions, Table 3 #1
TOTAL_TRUE_LABELS = 5143  # fixed denominator from paper


def simple_tokenize(text: str) -> List[str]:
    """Lowercase, split on whitespace/punctuation for BM25."""
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf_score(rankings: List[List[str]], k: int = 60) -> Dict[str, float]:
    """Reciprocal Rank Fusion over multiple ranked lists."""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, tid in enumerate(ranking, start=1):
            scores[tid] = scores.get(tid, 0.0) + 1.0 / (k + rank)
    return scores


def compute_recall_at_k(
    rankings: List[List[str]],
    ground_truth: List[Set[str]],
    ks: List[int] = [1, 5, 10, 20],
    total_labels: int = TOTAL_TRUE_LABELS,
) -> Dict[str, float]:
    """Recall@k with denominator = total_labels (5,143)."""
    results = {}
    for k in ks:
        hits = 0
        for top_cands, true_set in zip(rankings, ground_truth):
            overlap = true_set.intersection(set(top_cands[:k]))
            hits += len(overlap)
        results[f"recall@{k}"] = round(hits / total_labels * 100.0, 2)
        results[f"hits@{k}"] = hits
    return results


def main():
    print("=" * 70)
    print("EXTENSION: BM25 + RRF Fusion — not in paper (Schuurman 2025)")
    print("=" * 70)

    # 1. Load KB (50 techniques)
    print("\n[1/5] Loading 50-technique Knowledge Base...")
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb_data = json.load(f)
    techniques = kb_data["techniques"]
    technique_ids = [t["technique_id"] for t in techniques]
    tech_descs = {t["technique_id"]: t.get("description", "") for t in techniques}
    valid_50: Set[str] = set(technique_ids)
    print(f"  {len(techniques)} techniques loaded.")

    # 2. Load TRAM data (4,070 labelled sentences)
    print("\n[2/5] Loading TRAM data (labelled sentences)...")
    with open(TRAM_PATH, "r", encoding="utf-8") as f:
        tram_raw = json.load(f)

    labelled = [
        row for row in tram_raw
        if any(lbl in valid_50 for lbl in (row.get("labels") or []))
    ]
    sentence_texts = [row["sentence"] for row in labelled]
    ground_truth: List[Set[str]] = [
        {lbl for lbl in (row.get("labels") or []) if lbl in valid_50}
        for row in labelled
    ]
    total_labels_check = sum(len(g) for g in ground_truth)
    print(f"  {len(labelled)} labelled sentences, {total_labels_check} total label instances.")
    assert total_labels_check == TOTAL_TRUE_LABELS, (
        f"Expected {TOTAL_TRUE_LABELS} labels, got {total_labels_check}"
    )

    # 3. Load Nomic embeddings (pre-computed, primary = with prefixes)
    print("\n[3/5] Loading Nomic embeddings (with prefixes, primary setting)...")
    nomic_q = np.load(str(EMB_CACHE_DIR / "nomic_queries_with_prefixes.npy"))
    nomic_t = np.load(str(EMB_CACHE_DIR / "nomic_tech_full_with_prefixes.npy"))
    print(f"  Queries: {nomic_q.shape}, Techs: {nomic_t.shape}")

    # Normalize for cosine similarity
    nomic_q_norm = nomic_q / np.linalg.norm(nomic_q, axis=1, keepdims=True).clip(min=1e-9)
    nomic_t_norm = nomic_t / np.linalg.norm(nomic_t, axis=1, keepdims=True).clip(min=1e-9)

    # 4. Build BM25 index over technique descriptions
    print("\n[4/5] Building BM25 index over 50 technique descriptions...")
    t0 = time.time()
    corpus_tokens = [simple_tokenize(tech_descs[tid]) for tid in technique_ids]
    bm25 = BM25Okapi(corpus_tokens)
    bm25_build_time = round(time.time() - t0, 2)
    print(f"  BM25 index built in {bm25_build_time}s")

    # 5. Compute rankings for each method
    print("\n[5/5] Computing rankings for all 4,070 sentences...")

    # -- Nomic rankings (from pre-computed embeddings) --
    print("  Computing Nomic rankings...")
    t0 = time.time()
    sims = nomic_q_norm @ nomic_t_norm.T  # (4070, 50)
    nomic_rankings = []
    for i in range(len(labelled)):
        order = np.argsort(-sims[i])
        nomic_rankings.append([technique_ids[j] for j in order])
    nomic_time = round(time.time() - t0, 2)
    print(f"  Nomic rankings computed in {nomic_time}s")

    # -- BM25 rankings --
    print("  Computing BM25 rankings...")
    t0 = time.time()
    bm25_rankings = []
    for sent in sentence_texts:
        query_tokens = simple_tokenize(sent)
        scores = bm25.get_scores(query_tokens)
        order = np.argsort(-scores)
        bm25_rankings.append([technique_ids[j] for j in order])
    bm25_score_time = round(time.time() - t0, 2)
    print(f"  BM25 rankings computed in {bm25_score_time}s")

    # -- RRF fusion (BM25 + Nomic) --
    print("  Computing RRF fusion (k=60)...")
    t0 = time.time()
    rrf_rankings = []
    for i in range(len(labelled)):
        scores = rrf_score([bm25_rankings[i], nomic_rankings[i]], k=60)
        sorted_tids = sorted(scores.keys(), key=lambda tid: -scores[tid])
        rrf_rankings.append(sorted_tids)
    rrf_time = round(time.time() - t0, 2)
    print(f"  RRF fusion computed in {rrf_time}s")

    # 6. Compute Recall@k metrics
    print("\n  Computing Recall@k metrics...")
    nomic_metrics = compute_recall_at_k(nomic_rankings, ground_truth)
    bm25_metrics = compute_recall_at_k(bm25_rankings, ground_truth)
    rrf_metrics = compute_recall_at_k(rrf_rankings, ground_truth)

    # 7. Print results
    print()
    print("=" * 75)
    print("RESULTS (EXTENSION — not in paper)")
    print("=" * 75)
    print(f"{'Method':<30} | {'R@1':>6} | {'R@5':>6} | {'R@10':>7} | {'R@20':>7}")
    print("-" * 75)
    print(f"{'Paper: Nomic Full (target)':<30} | {'  ?':>6} | {'  ?':>6} | {PAPER_NOMIC_R10:>6.2f}% | {'  ?':>7}")
    print(f"{'Nomic w/ prefixes (repro)':<30} | {nomic_metrics['recall@1']:>5.2f}% | {nomic_metrics['recall@5']:>5.2f}% | {nomic_metrics['recall@10']:>6.2f}% | {nomic_metrics['recall@20']:>6.2f}%")
    print(f"{'BM25 (extension)':<30} | {bm25_metrics['recall@1']:>5.2f}% | {bm25_metrics['recall@5']:>5.2f}% | {bm25_metrics['recall@10']:>6.2f}% | {bm25_metrics['recall@20']:>6.2f}%")
    print(f"{'BM25 + Nomic RRF k=60 (ext)':<30} | {rrf_metrics['recall@1']:>5.2f}% | {rrf_metrics['recall@5']:>5.2f}% | {rrf_metrics['recall@10']:>6.2f}% | {rrf_metrics['recall@20']:>6.2f}%")
    print("=" * 75)
    print("Note: EXTENSION — BM25 and RRF not evaluated in the paper.")
    print("      Nomic row shown for comparison to paper baseline (80.28%).")

    # 8. Save results
    output = {
        "metadata": {
            "label": "EXTENSION — not in paper (Schuurman 2025)",
            "num_sentences": len(labelled),
            "total_true_labels": total_labels_check,
            "paper_nomic_r10": PAPER_NOMIC_R10,
            "bm25_tokenizer": "simple_tokenize (re.findall r'[a-z0-9]+', lowercase)",
            "rrf_k": 60,
            "nomic_embeddings": "nomic_queries_with_prefixes.npy + nomic_tech_full_with_prefixes.npy",
            "nomic_prefix": "search_query: (sentences), search_document: (tech descriptions)",
        },
        "results": {
            "nomic_with_prefixes": nomic_metrics,
            "bm25_alone": bm25_metrics,
            "bm25_nomic_rrf_k60": rrf_metrics,
        },
        "timings_seconds": {
            "bm25_build": bm25_build_time,
            "nomic_ranking": nomic_time,
            "bm25_scoring": bm25_score_time,
            "rrf_fusion": rrf_time,
        },
        "paper_target": {"nomic_full_r10": PAPER_NOMIC_R10},
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUT_FILE}")


if __name__ == "__main__":
    main()

