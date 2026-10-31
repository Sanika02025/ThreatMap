"""
investigate_table3_misses.py — Real empirical tests for Table 3 deviations.

Item 4c requirements:
- ATTACK-BERT (81.84 vs 86.05):
  Test mean pooling vs CLS, with and without text cleaning (strip "(Citation: ...)" and markdown links).
- Reranker (80.75 vs 74.59):
  Test max_length 256 vs 512, and descriptions with vs without cleaning.
Report every variant's Recall@10, whether or not it matches.
"""

import json
import re
import time
from pathlib import Path
from typing import List, Set

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
KB_PATH = BASE_DIR / "data" / "processed" / "mitre_attack_50_techniques.json"
TRAM_PATH = BASE_DIR / "data" / "tram2-data" / "multi_label.json"
EMB_CACHE_DIR = BASE_DIR / "artifacts" / "embedding_cache"
OUT_FILE = BASE_DIR / "artifacts" / "table3_ablation_investigation.json"

TOTAL_TRUE_LABELS = 5143


def clean_technique_text(text: str) -> str:
    """Strip '(Citation: ...)' and convert markdown links '[text](url)' to 'text'."""
    # Strip (Citation: ...)
    cleaned = re.sub(r"\(Citation:[^)]*\)", "", text)
    # Strip markdown links [anchor](url) -> anchor
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    # Strip extra whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def compute_recall_metrics(rankings: List[List[str]], ground_truth: List[Set[str]], total_labels: int = 5143):
    hits10 = 0
    for top_cands, true_set in zip(rankings, ground_truth):
        hits10 += len(true_set.intersection(set(top_cands[:10])))
    return round((hits10 / total_labels) * 100.0, 2), hits10


def main():
    print("=" * 80)
    print("ITEM 4c: Real Empirical Investigation of Table 3 Misses")
    print("=" * 80)

    # 1. Load KB & TRAM data
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb_data = json.load(f)
    techniques = kb_data["techniques"]
    technique_ids = [t["technique_id"] for t in techniques]
    valid_50 = set(technique_ids)

    with open(TRAM_PATH, "r", encoding="utf-8") as f:
        tram_raw = json.load(f)
    labelled = [r for r in tram_raw if any(lbl in valid_50 for lbl in (r.get("labels") or []))]
    sentence_texts = [r["sentence"].strip() for r in labelled]
    ground_truth = [{lbl for lbl in (r.get("labels") or []) if lbl in valid_50} for r in labelled]

    raw_descs = [t["description"] for t in techniques]
    cleaned_descs = [clean_technique_text(d) for d in raw_descs]

    ablation_results = {}

    # =========================================================================
    # Part 1: ATTACK-BERT Ablations
    # =========================================================================
    print("\n--- Part 1: ATTACK-BERT Pooling & Cleaning Ablations ---")
    attack_bert = SentenceTransformer("basel/ATTACK-BERT", local_files_only=True)
    attack_bert.max_seq_length = 512

    # Check pooling mode in current SentenceTransformer model
    # Usually modules[-1] is Pooling
    pooling_layer = None
    for m in attack_bert.modules():
        if hasattr(m, "pooling_mode_mean_tokens"):
            pooling_layer = m
            break

    # We test 4 variants:
    # (a) Mean pooling, raw descriptions
    # (b) Mean pooling, cleaned descriptions
    # (c) CLS pooling, raw descriptions
    # (d) CLS pooling, cleaned descriptions

    # Load pre-computed mean queries if available
    query_embs_mean = np.load(str(EMB_CACHE_DIR / "attack_bert_queries.npy"))

    # (a) Mean pooling, raw descriptions
    t_raw_mean = attack_bert.encode(raw_descs, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    sim_a = np.dot(query_embs_mean, t_raw_mean.T)
    rank_a = [[technique_ids[idx] for idx in np.argsort(-sim_a[i])] for i in range(len(sentence_texts))]
    r10_a, hits_a = compute_recall_metrics(rank_a, ground_truth)
    print(f"ATTACK-BERT (Mean pooling, Raw descriptions):     Recall@10 = {r10_a}% ({hits_a}/{TOTAL_TRUE_LABELS})")
    ablation_results["attack_bert_mean_raw"] = {"recall@10": r10_a, "hits": hits_a}

    # (b) Mean pooling, cleaned descriptions
    t_clean_mean = attack_bert.encode(cleaned_descs, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    sim_b = np.dot(query_embs_mean, t_clean_mean.T)
    rank_b = [[technique_ids[idx] for idx in np.argsort(-sim_b[i])] for i in range(len(sentence_texts))]
    r10_b, hits_b = compute_recall_metrics(rank_b, ground_truth)
    print(f"ATTACK-BERT (Mean pooling, Cleaned descriptions): Recall@10 = {r10_b}% ({hits_b}/{TOTAL_TRUE_LABELS})")
    ablation_results["attack_bert_mean_clean"] = {"recall@10": r10_b, "hits": hits_b}

    # If pooling layer allows switching to CLS:
    if pooling_layer is not None:
        print("\nSwitching ATTACK-BERT pooling layer to CLS...")
        orig_mean = pooling_layer.pooling_mode_mean_tokens
        orig_cls = pooling_layer.pooling_mode_cls_token

        pooling_layer.pooling_mode_mean_tokens = False
        pooling_layer.pooling_mode_cls_token = True

        # Check if we have cached CLS queries, or compute
        cls_q_cache = EMB_CACHE_DIR / "attack_bert_queries_cls.npy"
        if cls_q_cache.exists():
            print("Loading cached CLS queries...")
            query_embs_cls = np.load(str(cls_q_cache))
        else:
            print("Encoding 4,070 queries with CLS pooling...")
            query_embs_cls = attack_bert.encode(sentence_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
            np.save(str(cls_q_cache), query_embs_cls)

        # (c) CLS pooling, raw descriptions
        t_raw_cls = attack_bert.encode(raw_descs, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        sim_c = np.dot(query_embs_cls, t_raw_cls.T)
        rank_c = [[technique_ids[idx] for idx in np.argsort(-sim_c[i])] for i in range(len(sentence_texts))]
        r10_c, hits_c = compute_recall_metrics(rank_c, ground_truth)
        print(f"ATTACK-BERT (CLS pooling, Raw descriptions):      Recall@10 = {r10_c}% ({hits_c}/{TOTAL_TRUE_LABELS})")
        ablation_results["attack_bert_cls_raw"] = {"recall@10": r10_c, "hits": hits_c}

        # (d) CLS pooling, cleaned descriptions
        t_clean_cls = attack_bert.encode(cleaned_descs, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        sim_d = np.dot(query_embs_cls, t_clean_cls.T)
        rank_d = [[technique_ids[idx] for idx in np.argsort(-sim_d[i])] for i in range(len(sentence_texts))]
        r10_d, hits_d = compute_recall_metrics(rank_d, ground_truth)
        print(f"ATTACK-BERT (CLS pooling, Cleaned descriptions):  Recall@10 = {r10_d}% ({hits_d}/{TOTAL_TRUE_LABELS})")
        ablation_results["attack_bert_cls_clean"] = {"recall@10": r10_d, "hits": hits_d}

        # Restore pooling layer
        pooling_layer.pooling_mode_mean_tokens = orig_mean
        pooling_layer.pooling_mode_cls_token = orig_cls

    # =========================================================================
    # Part 2: Reranker Ablations (max_length 256 vs 512, raw vs cleaned)
    # =========================================================================
    print("\n--- Part 2: ms-marco Reranker Ablations ---")
    # Load baseline nomic rankings (with prefixes)
    nomic_q = np.load(str(EMB_CACHE_DIR / "nomic_queries_with_prefixes.npy"))
    nomic_t = np.load(str(EMB_CACHE_DIR / "nomic_tech_full_with_prefixes.npy"))
    nomic_sim = np.dot(nomic_q, nomic_t.T)
    baseline_nomic_rankings = [[technique_ids[idx] for idx in np.argsort(-nomic_sim[i])] for i in range(len(sentence_texts))]

    reranker_variants = [
        ("reranker_max256_raw", 256, False),
        ("reranker_max512_raw", 512, False),
        ("reranker_max256_clean", 256, True),
        ("reranker_max512_clean", 512, True),
    ]

    for var_id, max_len, use_clean in reranker_variants:
        desc_lookup = {t["technique_id"]: (clean_technique_text(t["description"]) if use_clean else t["description"]) for t in techniques}
        
        cache_name = f"msmarco_reranker_scores_{var_id}"
        cache_p = EMB_CACHE_DIR / f"{cache_name}.npy"

        if cache_p.exists():
            print(f"Loading cached {var_id} scores...")
            ce_scores = np.load(str(cache_p))
        else:
            print(f"Running CrossEncoder ({var_id}: max_len={max_len}, clean={use_clean})...")
            ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=max_len, local_files_only=True)
            all_pairs = []
            for s_idx, s_text in enumerate(sentence_texts):
                for tid in baseline_nomic_rankings[s_idx][:15]:
                    all_pairs.append([s_text, desc_lookup[tid]])
            ce_scores = ce.predict(all_pairs, batch_size=256, show_progress_bar=False)
            np.save(str(cache_p), ce_scores)

        # Build reranked rankings
        curr = 0
        reranked = []
        for s_idx in range(len(sentence_texts)):
            top_15 = baseline_nomic_rankings[s_idx][:15]
            sc = ce_scores[curr : curr + len(top_15)]
            curr += len(top_15)
            sorted_15 = [top_15[idx] for idx in np.argsort(-sc)]
            reranked.append(sorted_15 + baseline_nomic_rankings[s_idx][15:])

        r10, hits = compute_recall_metrics(reranked, ground_truth)
        clean_lbl = "Clean" if use_clean else "Raw"
        print(f"Reranker (max_len={max_len}, {clean_lbl}): Recall@10 = {r10}% ({hits}/{TOTAL_TRUE_LABELS})")
        ablation_results[var_id] = {"recall@10": r10, "hits": hits, "max_length": max_len, "cleaned": use_clean}

    print("\n" + "=" * 80)
    print("SUMMARY OF EMPIRICAL ABLATION EXPERIMENTS")
    print("=" * 80)
    print(f"Paper Target ATTACK-BERT Full: 86.05%")
    for k in ["attack_bert_mean_raw", "attack_bert_mean_clean", "attack_bert_cls_raw", "attack_bert_cls_clean"]:
        if k in ablation_results:
            print(f"  {k:<30}: {ablation_results[k]['recall@10']}%")
    print(f"\nPaper Target ms-marco Reranker: 74.59% (hurts vs baseline nomic 80.28%)")
    for k in ["reranker_max256_raw", "reranker_max512_raw", "reranker_max256_clean", "reranker_max512_clean"]:
        if k in ablation_results:
            print(f"  {k:<30}: {ablation_results[k]['recall@10']}%")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\nSaved ablation investigation results to {OUT_FILE}")


if __name__ == "__main__":
    main()

