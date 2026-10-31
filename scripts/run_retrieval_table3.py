"""
Real, non-simulated empirical reproduction of Table 3 from RAG-ATT&CK (Schuurman, 2025).

Hard Rules:
- No np.random / random / hardcoded simulation results.
- Embeddings are computed with real models over the 4,070 labelled sentences of TRAM v1.3.
- Evaluates against the 50 technique classes with fixed denominator = 5,143 true labels.
"""

import json
import os
from pathlib import Path
import time
from typing import Dict, List, Set, Tuple

import numpy as np

# Ensure offline loading from cached model weights
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_XET_DISABLED"] = "1"

BASE_DIR = Path(__file__).resolve().parent.parent

# Pinned TRAM commit & paths
TRAM_PINNED_COMMIT = "f29793d8d665f7f552898696e00065ef24a29a20"
TRAM_PATH = BASE_DIR / "data" / "tram2-data" / "multi_label.json"
if not TRAM_PATH.exists():
    TRAM_PATH = BASE_DIR / "data" / "raw" / "tram_multi_label.json"

KB_PATH = BASE_DIR / "data" / "processed" / "mitre_attack_50_techniques.json"
EMB_CACHE_DIR = BASE_DIR / "artifacts" / "embedding_cache"
EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_or_compute_embeddings(cache_name: str, compute_fn) -> np.ndarray:
    cache_path = EMB_CACHE_DIR / f"{cache_name}.npy"
    if cache_path.exists():
        print(f"Loading cached embeddings from {cache_path.name}...")
        return np.load(cache_path)
    embs = compute_fn()
    np.save(cache_path, embs)
    return embs

# Exact paper targets from RAG-ATT&CK Table 3
PAPER_TARGETS = {
    "nomic_full": {
        "name": "Nomic Full Descriptions",
        "r1": 37.44,
        "r5": 69.06,
        "r10": 80.28,
        "r20": 89.08,
    },
    "nomic_first_paragraph": {
        "name": "Nomic First Paragraph",
        "r1": None,
        "r5": None,
        "r10": 77.06,
        "r20": None,
    },
    "nomic_summary": {
        "name": "Nomic Summarized Descriptions",
        "r1": None,
        "r5": None,
        "r10": 83.93,
        "r20": None,
    },
    "nomic_chunk_750": {
        "name": "Nomic Chunks 750/80",
        "r1": None,
        "r5": None,
        "r10": 77.61,
        "r20": None,
    },
    "nomic_chunk_375": {
        "name": "Nomic Chunks 375/50",
        "r1": None,
        "r5": None,
        "r10": 71.70,
        "r20": None,
    },
    "attack_bert_full": {
        "name": "ATTACK-BERT Full",
        "r1": 42.22,
        "r5": 75.66,
        "r10": 86.05,
        "r20": 94.08,
    },
    "reranker_top15": {
        "name": "ms-marco Reranker (Top 15)",
        "r1": None,
        "r5": None,
        "r10": 74.59,
        "r20": None,
    },
}


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Splits text into sliding window chunks."""
    text = text.strip()
    if not text:
        return [""]
    if len(text) <= chunk_size:
        return [text]
    step = max(1, chunk_size - overlap)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks if chunks else [text]


def compute_recall_metrics(
    rankings: List[List[str]],
    ground_truth: List[Set[str]],
    total_labels: int = 5143,
) -> Dict[str, float]:
    """
    Computes Recall@k (k=1, 5, 10, 20) with denominator fixed to 5,143 true labels.
    """
    metrics = {}
    for k in [1, 5, 10, 20]:
        hits = 0
        for top_cands, true_set in zip(rankings, ground_truth):
            overlap = true_set.intersection(set(top_cands[:k]))
            hits += len(overlap)
        recall = round((hits / total_labels) * 100.0, 2)
        metrics[f"recall@{k}"] = recall
        metrics[f"hits@{k}"] = hits
    return metrics


def run_real_table3_evaluation():
    print("=" * 80)
    print("REAL EMPIRICAL EVALUATION: RAG-ATT&CK Table 3 Reproduction")
    print(f"Pinned TRAM commit: {TRAM_PINNED_COMMIT}")
    print("=" * 80)

    # 1. Load Knowledge Base (50 techniques)
    print("\n[Step 1] Loading 50 techniques Knowledge Base...")
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb_data = json.load(f)
    techniques = kb_data["techniques"]
    technique_ids = [t["technique_id"] for t in techniques]
    valid_50 = set(technique_ids)
    print(f"Loaded {len(techniques)} technique definitions.")

    # 2. Load TRAM dataset and filter to 4,070 labelled sentences
    print("\n[Step 2] Loading TRAM dataset...")
    with open(TRAM_PATH, "r", encoding="utf-8") as f:
        tram_raw = json.load(f)
    print(f"Total rows in raw file: {len(tram_raw)}")

    # Keep ONLY rows with labels in the 50 classes
    labeled_rows = []
    ground_truth = []
    sentence_texts = []
    for r in tram_raw:
        true_in_scope = set(l for l in r.get("labels", []) if l in valid_50)
        if true_in_scope:
            labeled_rows.append(r)
            ground_truth.append(true_in_scope)
            sentence_texts.append(r.get("sentence", "").strip())

    total_true_labels = sum(len(gt) for gt in ground_truth)
    print(f"Filtered labelled sentences: {len(sentence_texts)} (Expected: 4,070)")
    print(f"Total true label instances: {total_true_labels} (Expected: 5,143)")
    assert len(sentence_texts) == 4070, f"Expected 4070 sentences, got {len(sentence_texts)}"
    assert total_true_labels == 5143, f"Expected 5143 labels, got {total_true_labels}"

    results = {}
    timings = {}

    from sentence_transformers import CrossEncoder, SentenceTransformer

    # -------------------------------------------------------------------------
    # 3. Model: nomic-ai/nomic-embed-text-v1.5
    # -------------------------------------------------------------------------
    print("\n[Step 3] Loading nomic-ai/nomic-embed-text-v1.5...")
    nomic_model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        local_files_only=True,
    )
    nomic_model.max_seq_length = 512

    for use_prefixes in [True, False]:
        prefix_label = "with_prefixes" if use_prefixes else "no_prefixes"
        doc_pfx = "search_document: " if use_prefixes else ""
        query_pfx = "search_query: " if use_prefixes else ""
        print(f"\n--- Running Nomic ({prefix_label}) ---")

        # Encode queries (4,070 sentences)
        t0 = time.time()
        queries = [query_pfx + s for s in sentence_texts]
        print(f"Encoding {len(queries)} sentence queries...")
        query_embs = get_or_compute_embeddings(
            f"nomic_queries_{prefix_label}",
            lambda: nomic_model.encode(queries, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        )
        t_queries = round(time.time() - t0, 2)
        print(f"Queries ready in {t_queries}s.")

        # Variant A: Full Descriptions
        t0 = time.time()
        docs_full = [doc_pfx + t["description"] for t in techniques]
        tech_embs_full = get_or_compute_embeddings(
            f"nomic_tech_full_{prefix_label}",
            lambda: nomic_model.encode(docs_full, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        )
        sim_full = np.dot(query_embs, tech_embs_full.T)  # (4070, 50)
        rankings_full = [[technique_ids[idx] for idx in np.argsort(-sim_full[i])] for i in range(len(sentence_texts))]
        metrics_full = compute_recall_metrics(rankings_full, ground_truth, total_labels=5143)
        t_full = round(time.time() - t0 + t_queries, 2)
        results[f"nomic_full_{prefix_label}"] = metrics_full
        timings[f"nomic_full_{prefix_label}"] = t_full
        print(f"Nomic Full ({prefix_label}): Recall@10 = {metrics_full['recall@10']}% ({metrics_full['hits@10']}/5143)")

        # Save baseline rankings for reranker evaluation
        if use_prefixes:
            baseline_nomic_rankings = rankings_full
            baseline_nomic_sim = sim_full

        # Variant B: First Paragraph (split at first blank line)
        t0 = time.time()
        docs_para = [doc_pfx + t["description"].split("\n\n")[0].strip() for t in techniques]
        tech_embs_para = get_or_compute_embeddings(
            f"nomic_tech_para_{prefix_label}",
            lambda: nomic_model.encode(docs_para, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        )
        sim_para = np.dot(query_embs, tech_embs_para.T)
        rankings_para = [[technique_ids[idx] for idx in np.argsort(-sim_para[i])] for i in range(len(sentence_texts))]
        metrics_para = compute_recall_metrics(rankings_para, ground_truth, total_labels=5143)
        t_para = round(time.time() - t0 + t_queries, 2)
        results[f"nomic_first_paragraph_{prefix_label}"] = metrics_para
        timings[f"nomic_first_paragraph_{prefix_label}"] = t_para
        print(f"Nomic First Paragraph ({prefix_label}): Recall@10 = {metrics_para['recall@10']}%")

        # Variant C: Chunks 750/80
        t0 = time.time()
        all_chunks_750 = []
        chunk_to_tech_750 = []
        for tech_idx, t in enumerate(techniques):
            chunks = chunk_text(t["description"], chunk_size=750, overlap=80)
            for c in chunks:
                all_chunks_750.append(doc_pfx + c)
                chunk_to_tech_750.append(tech_idx)
        
        chunk_embs_750 = get_or_compute_embeddings(
            f"nomic_chunk_750_{prefix_label}",
            lambda: nomic_model.encode(all_chunks_750, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        )
        sim_chunk_750 = np.dot(query_embs, chunk_embs_750.T)  # (4070, num_chunks)
        
        # technique score = max chunk score
        sim_tech_750 = np.full((len(sentence_texts), 50), -1.0, dtype=np.float32)
        for chunk_idx, tech_idx in enumerate(chunk_to_tech_750):
            sim_tech_750[:, tech_idx] = np.maximum(sim_tech_750[:, tech_idx], sim_chunk_750[:, chunk_idx])
            
        rankings_750 = [[technique_ids[idx] for idx in np.argsort(-sim_tech_750[i])] for i in range(len(sentence_texts))]
        metrics_750 = compute_recall_metrics(rankings_750, ground_truth, total_labels=5143)
        t_750 = round(time.time() - t0 + t_queries, 2)
        results[f"nomic_chunk_750_{prefix_label}"] = metrics_750
        timings[f"nomic_chunk_750_{prefix_label}"] = t_750
        print(f"Nomic Chunks 750/80 ({prefix_label}): Recall@10 = {metrics_750['recall@10']}%")

        # Variant D: Chunks 375/50
        t0 = time.time()
        all_chunks_375 = []
        chunk_to_tech_375 = []
        for tech_idx, t in enumerate(techniques):
            chunks = chunk_text(t["description"], chunk_size=375, overlap=50)
            for c in chunks:
                all_chunks_375.append(doc_pfx + c)
                chunk_to_tech_375.append(tech_idx)
        
        chunk_embs_375 = get_or_compute_embeddings(
            f"nomic_chunk_375_{prefix_label}",
            lambda: nomic_model.encode(all_chunks_375, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        )
        sim_chunk_375 = np.dot(query_embs, chunk_embs_375.T)
        
        sim_tech_375 = np.full((len(sentence_texts), 50), -1.0, dtype=np.float32)
        for chunk_idx, tech_idx in enumerate(chunk_to_tech_375):
            sim_tech_375[:, tech_idx] = np.maximum(sim_tech_375[:, tech_idx], sim_chunk_375[:, chunk_idx])
            
        rankings_375 = [[technique_ids[idx] for idx in np.argsort(-sim_tech_375[i])] for i in range(len(sentence_texts))]
        metrics_375 = compute_recall_metrics(rankings_375, ground_truth, total_labels=5143)
        t_375 = round(time.time() - t0 + t_queries, 2)
        results[f"nomic_chunk_375_{prefix_label}"] = metrics_375
        timings[f"nomic_chunk_375_{prefix_label}"] = t_375
        print(f"Nomic Chunks 375/50 ({prefix_label}): Recall@10 = {metrics_375['recall@10']}%")

    # -------------------------------------------------------------------------
    # 4. Model: basel/ATTACK-BERT
    # -------------------------------------------------------------------------
    print("\n[Step 4] Loading basel/ATTACK-BERT...")
    t0 = time.time()
    attack_bert = SentenceTransformer("basel/ATTACK-BERT", local_files_only=True)
    attack_bert.max_seq_length = 512
    queries_bert = sentence_texts
    query_embs_bert = get_or_compute_embeddings(
        "attack_bert_queries",
        lambda: attack_bert.encode(queries_bert, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    )
    docs_bert = [t["description"] for t in techniques]
    tech_embs_bert = get_or_compute_embeddings(
        "attack_bert_tech_full",
        lambda: attack_bert.encode(docs_bert, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    )
    sim_bert = np.dot(query_embs_bert, tech_embs_bert.T)
    rankings_bert = [[technique_ids[idx] for idx in np.argsort(-sim_bert[i])] for i in range(len(sentence_texts))]
    metrics_bert = compute_recall_metrics(rankings_bert, ground_truth, total_labels=5143)
    t_bert = round(time.time() - t0, 2)
    results["attack_bert_full"] = metrics_bert
    timings["attack_bert_full"] = t_bert
    print(f"ATTACK-BERT Full: Recall@10 = {metrics_bert['recall@10']}% ({metrics_bert['hits@10']}/5143)")

    # -------------------------------------------------------------------------
    # 5. Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 over Top 15
    # -------------------------------------------------------------------------
    print("\n[Step 5] Running ms-marco-MiniLM-L-6-v2 Reranker over Top 15...")
    t0 = time.time()
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", local_files_only=True)
    reranker.max_length = 256
    tech_desc_map = {t["technique_id"]: t["description"] for t in techniques}

    # Build pairs for top 15 candidates from baseline nomic
    reranked_rankings = []
    all_pairs = []
    pair_slices = []
    curr_idx = 0
    for s_idx, s_text in enumerate(sentence_texts):
        top_15 = baseline_nomic_rankings[s_idx][:15]
        for tid in top_15:
            all_pairs.append([s_text, tech_desc_map[tid]])
        pair_slices.append((curr_idx, curr_idx + len(top_15)))
        curr_idx += len(top_15)

    print(f"Scoring {len(all_pairs)} (sentence, candidate) pairs with CrossEncoder...")
    ce_scores = get_or_compute_embeddings(
        "msmarco_reranker_top15_scores",
        lambda: reranker.predict(all_pairs, batch_size=256, show_progress_bar=False)
    )

    for s_idx, (start_p, end_p) in enumerate(pair_slices):
        top_15 = baseline_nomic_rankings[s_idx][:15]
        scores_15 = ce_scores[start_p:end_p]
        # Re-sort top 15 in descending order of cross-encoder score
        sorted_top_15 = [top_15[idx] for idx in np.argsort(-scores_15)]
        # Append remaining candidates (ranks 16-50) from baseline
        full_reranked = sorted_top_15 + baseline_nomic_rankings[s_idx][15:]
        reranked_rankings.append(full_reranked)

    metrics_reranker = compute_recall_metrics(reranked_rankings, ground_truth, total_labels=5143)
    t_reranker = round(time.time() - t0, 2)
    results["reranker_top15"] = metrics_reranker
    timings["reranker_top15"] = t_reranker
    print(f"Reranker Top 15: Recall@10 = {metrics_reranker['recall@10']}% ({metrics_reranker['hits@10']}/5143)")

    # -------------------------------------------------------------------------
    # 6. Format and Print Comparison Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("TABLE 3 REPRODUCTION SUMMARY (Recall@10 & Ablations)")
    print("=" * 90)
    print(f"{'Experiment':<32} | {'Paper Target':<12} | {'Real Run':<10} | {'Delta':<8} | {'Status'}")
    print("-" * 90)

    comparisons = [
        ("Nomic Full (with prefixes)", PAPER_TARGETS["nomic_full"]["r10"], results["nomic_full_with_prefixes"]["recall@10"]),
        ("Nomic Full (no prefixes)", PAPER_TARGETS["nomic_full"]["r10"], results["nomic_full_no_prefixes"]["recall@10"]),
        ("Nomic First Paragraph", PAPER_TARGETS["nomic_first_paragraph"]["r10"], results["nomic_first_paragraph_with_prefixes"]["recall@10"]),
        ("Nomic Chunks 750/80", PAPER_TARGETS["nomic_chunk_750"]["r10"], results["nomic_chunk_750_with_prefixes"]["recall@10"]),
        ("Nomic Chunks 375/50", PAPER_TARGETS["nomic_chunk_375"]["r10"], results["nomic_chunk_375_with_prefixes"]["recall@10"]),
        ("ATTACK-BERT Full", PAPER_TARGETS["attack_bert_full"]["r10"], results["attack_bert_full"]["recall@10"]),
        ("ms-marco Reranker (Top 15)", PAPER_TARGETS["reranker_top15"]["r10"], results["reranker_top15"]["recall@10"]),
    ]

    for label, target, real_val in comparisons:
        delta = round(real_val - target, 2)
        status = "MATCH (<=3%)" if abs(delta) <= 3.0 else "DEVIATION"
        delta_str = f"+{delta}%" if delta > 0 else f"{delta}%"
        print(f"{label:<32} | {target:>10.2f}% | {real_val:>8.2f}% | {delta_str:>8} | {status}")

    print("=" * 90)

    # Detailed R@1, R@5, R@10, R@20 Breakdown
    print("\nDetailed Multi-k Recall Breakdown:")
    print(f"{'Experiment':<32} | {'R@1':<8} | {'R@5':<8} | {'R@10':<8} | {'R@20':<8} | {'Runtime (s)'}")
    print("-" * 80)
    for k_exp in ["nomic_full_with_prefixes", "nomic_full_no_prefixes", "attack_bert_full", "reranker_top15"]:
        m = results[k_exp]
        t = timings[k_exp]
        print(f"{k_exp:<32} | {m['recall@1']:>6.2f}% | {m['recall@5']:>6.2f}% | {m['recall@10']:>6.2f}% | {m['recall@20']:>6.2f}% | {t:>8.2f}s")

    # 7. Save real output JSON
    output_payload = {
        "metadata": {
            "dataset_commit": TRAM_PINNED_COMMIT,
            "dataset_file": str(TRAM_PATH),
            "num_sentences_embedded": len(sentence_texts),
            "num_true_labels": total_true_labels,
            "num_classes": len(techniques),
            "models": {
                "nomic": "nomic-ai/nomic-embed-text-v1.5",
                "attack_bert": "basel/ATTACK-BERT",
                "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "paper_targets": PAPER_TARGETS,
        "empirical_results": results,
        "timings_seconds": timings,
    }

    out_file = BASE_DIR / "artifacts" / "retrieval_table3_real_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nSaved empirical results to {out_file}")

    return output_payload


if __name__ == "__main__":
    run_real_table3_evaluation()
