"""
build_ablation_results.py — Assemble table3_ablation_investigation.json from
empirical results already computed and cached by investigate_table3_misses.py.

Only numbers that were actually computed are written. Variants that were not
run (max_length=512 reranker, cleaned rerankers, CLS pooling) are marked
NOT_RUN with accurate per-variant explanations.
"""

import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
KB_PATH = BASE_DIR / "data" / "processed" / "mitre_attack_50_techniques.json"
TRAM_PATH = BASE_DIR / "data" / "tram2-data" / "multi_label.json"
EMB_CACHE_DIR = BASE_DIR / "artifacts" / "embedding_cache"
OUT_FILE = BASE_DIR / "artifacts" / "table3_ablation_investigation.json"

TOTAL_TRUE_LABELS = 5143


def compute_recall_metrics(rankings, ground_truth, total_labels=TOTAL_TRUE_LABELS):
    hits10 = 0
    for top_cands, true_set in zip(rankings, ground_truth):
        hits10 += len(true_set.intersection(set(top_cands[:10])))
    return round((hits10 / total_labels) * 100.0, 2), hits10


def main():
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

    results = {}

    # ── ATTACK-BERT Mean pooling, raw ────────────────────────────────────────
    print("Computing ATTACK-BERT mean/raw from cached embeddings...")
    q_embs = np.load(str(EMB_CACHE_DIR / "attack_bert_queries.npy"))
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("basel/ATTACK-BERT", local_files_only=True)
    model.max_seq_length = 512

    import re
    def clean(text):
        t = re.sub(r"\(Citation:[^)]*\)", "", text)
        t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
        return re.sub(r"\s+", " ", t).strip()

    raw_descs = [t["description"] for t in techniques]
    cleaned_descs = [clean(d) for d in raw_descs]

    t_raw = model.encode(raw_descs, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    sims = np.dot(q_embs, t_raw.T)
    rank_a = [[technique_ids[idx] for idx in np.argsort(-sims[i])] for i in range(len(sentence_texts))]
    r10_a, hits_a = compute_recall_metrics(rank_a, ground_truth)
    print(f"  Mean pooling, Raw:    Recall@10 = {r10_a}% ({hits_a}/{TOTAL_TRUE_LABELS})")
    results["attack_bert_mean_raw"] = {
        "recall@10": r10_a, "hits": hits_a,
        "pooling": "mean", "cleaned": False, "status": "DONE"
    }

    # ── ATTACK-BERT Mean pooling, cleaned ────────────────────────────────────
    t_clean = model.encode(cleaned_descs, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    sims_c = np.dot(q_embs, t_clean.T)
    rank_b = [[technique_ids[idx] for idx in np.argsort(-sims_c[i])] for i in range(len(sentence_texts))]
    r10_b, hits_b = compute_recall_metrics(rank_b, ground_truth)
    print(f"  Mean pooling, Cleaned: Recall@10 = {r10_b}% ({hits_b}/{TOTAL_TRUE_LABELS})")
    results["attack_bert_mean_clean"] = {
        "recall@10": r10_b, "hits": hits_b,
        "pooling": "mean", "cleaned": True, "status": "DONE"
    }

    # ── ATTACK-BERT CLS pooling — NOT RUN (time/hardware constraints) ────────
    results["attack_bert_cls_raw"] = {
        "recall@10": None, "hits": None,
        "pooling": "cls", "cleaned": False,
        "status": "NOT_RUN",
        "reason": "Not run (time/hardware constraints on CPU)."
    }
    results["attack_bert_cls_clean"] = {
        "recall@10": None, "hits": None,
        "pooling": "cls", "cleaned": True,
        "status": "NOT_RUN",
        "reason": "Not run (time/hardware constraints on CPU)."
    }

    # ── Reranker max_length=256, raw ─────────────────────────────────────────
    nomic_q = np.load(str(EMB_CACHE_DIR / "nomic_queries_with_prefixes.npy"))
    nomic_t = np.load(str(EMB_CACHE_DIR / "nomic_tech_full_with_prefixes.npy"))
    nomic_sim = np.dot(nomic_q, nomic_t.T)
    baseline_nomic_rankings = [
        [technique_ids[idx] for idx in np.argsort(-nomic_sim[i])]
        for i in range(len(sentence_texts))
    ]

    cache_256_raw = EMB_CACHE_DIR / "msmarco_reranker_scores_reranker_max256_raw.npy"
    ce_scores_256 = np.load(str(cache_256_raw))
    curr = 0
    reranked_256_raw = []
    for s_idx in range(len(sentence_texts)):
        top15 = baseline_nomic_rankings[s_idx][:15]
        sc = ce_scores_256[curr: curr + len(top15)]
        curr += len(top15)
        sorted15 = [top15[i] for i in np.argsort(-sc)]
        reranked_256_raw.append(sorted15 + baseline_nomic_rankings[s_idx][15:])
    r10_r256, hits_r256 = compute_recall_metrics(reranked_256_raw, ground_truth)
    print(f"  Reranker max_length=256, Raw:  Recall@10 = {r10_r256}% ({hits_r256}/{TOTAL_TRUE_LABELS})")
    results["reranker_max256_raw"] = {
        "recall@10": r10_r256, "hits": hits_r256,
        "max_length": 256, "cleaned": False, "status": "DONE"
    }

    # ── Reranker max_length=512 and clean variants — NOT RUN ─────────────────
    results["reranker_max512_raw"] = {
        "recall@10": None, "hits": None,
        "max_length": 512, "cleaned": False,
        "status": "NOT_RUN",
        "reason": "Not run (CrossEncoder inference with max_length=512 hung after model load on CPU)."
    }
    results["reranker_max256_clean"] = {
        "recall@10": None, "hits": None,
        "max_length": 256, "cleaned": True,
        "status": "NOT_RUN",
        "reason": "Not run (time/hardware constraints on CPU)."
    }
    results["reranker_max512_clean"] = {
        "recall@10": None, "hits": None,
        "max_length": 512, "cleaned": True,
        "status": "NOT_RUN",
        "reason": "Not run (time/hardware constraints on CPU)."
    }

    # Write
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ablation results to {OUT_FILE}")


if __name__ == "__main__":
    main()
