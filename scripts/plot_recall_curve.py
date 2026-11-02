"""
plot_recall_curve.py — Recall@k curves for k=1..50.

Models: nomic (with prefixes), ATTACK-BERT, BM25, BM25+nomic RRF k=60.
All rankings recomputed from cached embeddings and BM25 (same logic as
run_retrieval_table3.py and run_bm25_rrf.py; no hand-copying of results).

Outputs:
  - artifacts/fig_recall_at_k.png
  - artifacts/recall_at_k.json

Hard rules:
  - No simulation, no random number generation.
  - Paper R@10 reference markers for nomic (80.28%) and ATTACK-BERT (86.05%).
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRAM_PATH = ROOT / "data" / "tram2-data" / "multi_label.json"
KB_PATH   = ROOT / "data" / "processed" / "mitre_attack_50_techniques.json"
EMB_DIR   = ROOT / "artifacts" / "embedding_cache"
OUT_PNG   = ROOT / "artifacts" / "fig_recall_at_k.png"
OUT_JSON  = ROOT / "artifacts" / "recall_at_k.json"

TOTAL_LABELS = 5143
KS = list(range(1, 51))  # k = 1..50

# Paper reference markers at k=10
PAPER_REFS = {
    "nomic":        80.28,
    "attack_bert":  86.05,
}


def simple_tokenize(text: str) -> List[str]:
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf_score(rankings: List[List[str]], k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, tid in enumerate(ranking, start=1):
            scores[tid] = scores.get(tid, 0.0) + 1.0 / (k + rank)
    return scores


def recall_at_k_series(
    rankings: List[List[str]],
    ground_truth: List[Set[str]],
    ks: List[int],
    total_labels: int,
) -> List[float]:
    """Returns recall for each k in ks (sorted asc)."""
    # Sort ks, then compute hits for each k
    results = []
    for k in ks:
        hits = 0
        for top, gold in zip(rankings, ground_truth):
            hits += len(gold.intersection(set(top[:k])))
        results.append(round(hits / total_labels * 100.0, 4))
    return results


def cosine_rankings(q_embs: np.ndarray, d_embs: np.ndarray, tids: List[str]) -> List[List[str]]:
    q_norm = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-10)
    d_norm = d_embs / (np.linalg.norm(d_embs, axis=1, keepdims=True) + 1e-10)
    sims = q_norm @ d_norm.T
    rankings = []
    for row in sims:
        order = np.argsort(-row)
        rankings.append([tids[i] for i in order])
    return rankings


def main():
    print("=" * 65)
    print("Recall@k Curves (k=1..50)")
    print("=" * 65)

    # Load KB
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)
    techniques = kb["techniques"]
    technique_ids = [t["technique_id"] for t in techniques]
    valid_50 = set(technique_ids)
    tech_descs = {t["technique_id"]: t.get("description", "") for t in techniques}

    # Load TRAM
    with open(TRAM_PATH, "r", encoding="utf-8") as f:
        tram = json.load(f)
    labelled = [r for r in tram if any(l in valid_50 for l in (r.get("labels") or []))]
    ground_truth = [{l for l in (r.get("labels") or []) if l in valid_50} for r in labelled]
    sentence_texts = [r["sentence"] for r in labelled]
    assert sum(len(g) for g in ground_truth) == TOTAL_LABELS, "Label count mismatch!"
    print(f"Loaded {len(labelled)} sentences, {TOTAL_LABELS} label instances.")

    curves = {}

    # -- Nomic with prefixes --
    nomic_q = np.load(str(EMB_DIR / "nomic_queries_with_prefixes.npy"))
    nomic_d = np.load(str(EMB_DIR / "nomic_tech_full_with_prefixes.npy"))
    print("Computing nomic rankings...")
    nomic_rankings = cosine_rankings(nomic_q, nomic_d, technique_ids)
    curves["nomic"] = recall_at_k_series(nomic_rankings, ground_truth, KS, TOTAL_LABELS)
    print(f"  Nomic R@10 = {curves['nomic'][9]:.2f}%  (paper: {PAPER_REFS['nomic']}%)")

    # -- ATTACK-BERT --
    ab_q = np.load(str(EMB_DIR / "attack_bert_queries.npy"))
    ab_d = np.load(str(EMB_DIR / "attack_bert_tech_full.npy"))
    print("Computing ATTACK-BERT rankings...")
    ab_rankings = cosine_rankings(ab_q, ab_d, technique_ids)
    curves["attack_bert"] = recall_at_k_series(ab_rankings, ground_truth, KS, TOTAL_LABELS)
    print(f"  ATTACK-BERT R@10 = {curves['attack_bert'][9]:.2f}%  (paper: {PAPER_REFS['attack_bert']}%)")

    # -- BM25 --
    try:
        from rank_bm25 import BM25Okapi
        print("Building BM25 index...")
        corpus_tokens = [simple_tokenize(tech_descs[tid]) for tid in technique_ids]
        bm25 = BM25Okapi(corpus_tokens)
        bm25_rankings = []
        for sent in sentence_texts:
            scores = bm25.get_scores(simple_tokenize(sent))
            order = np.argsort(-np.array(scores))
            bm25_rankings.append([technique_ids[j] for j in order])
        curves["bm25"] = recall_at_k_series(bm25_rankings, ground_truth, KS, TOTAL_LABELS)
        print(f"  BM25 R@10 = {curves['bm25'][9]:.2f}%")

        # -- RRF --
        print("Computing RRF rankings...")
        rrf_rankings = []
        for i in range(len(labelled)):
            sc = rrf_score([bm25_rankings[i], nomic_rankings[i]], k=60)
            rrf_rankings.append(sorted(sc.keys(), key=lambda tid: -sc[tid]))
        curves["bm25_nomic_rrf"] = recall_at_k_series(rrf_rankings, ground_truth, KS, TOTAL_LABELS)
        print(f"  BM25+nomic RRF R@10 = {curves['bm25_nomic_rrf'][9]:.2f}%")
    except ImportError:
        print("rank_bm25 not installed — BM25 and RRF skipped.")
        curves["bm25"] = None
        curves["bm25_nomic_rrf"] = None

    # -- Save JSON --
    out_data = {
        "description": (
            "Recall@k for k=1..50 over 5,143 labels (4,070 sentences). "
            "Rankings recomputed from cached embeddings and BM25. No simulation."
        ),
        "ks": KS,
        "total_labels": TOTAL_LABELS,
        "paper_r10_references": PAPER_REFS,
        "curves": curves,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved JSON -> {OUT_JSON}")

    # -- Plot --
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(9, 6))

        colors = {
            "nomic": "#1f77b4",
            "attack_bert": "#d62728",
            "bm25": "#ff7f0e",
            "bm25_nomic_rrf": "#2ca02c",
        }
        labels_map = {
            "nomic": "Nomic (with prefixes) — reproduced",
            "attack_bert": "ATTACK-BERT — reproduced",
            "bm25": "BM25 (extension)",
            "bm25_nomic_rrf": "BM25 + Nomic RRF k=60 (extension)",
        }

        for key, vals in curves.items():
            if vals is None:
                continue
            ax.plot(KS, vals, color=colors[key], linewidth=2, label=labels_map[key])

        # Paper reference markers at k=10
        ax.scatter([10], [PAPER_REFS["nomic"]], color=colors["nomic"],
                   marker="x", s=120, zorder=5,
                   label=f"Paper: Nomic R@10 = {PAPER_REFS['nomic']}%")
        ax.scatter([10], [PAPER_REFS["attack_bert"]], color=colors["attack_bert"],
                   marker="x", s=120, zorder=5,
                   label=f"Paper: ATTACK-BERT R@10 = {PAPER_REFS['attack_bert']}%")

        ax.axvline(10, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("k", fontsize=13)
        ax.set_ylabel("Recall@k (%)", fontsize=13)
        ax.set_title("Recall@k (k = 1..50) over 5,143 label instances", fontsize=14)
        ax.legend(fontsize=9, loc="lower right")
        ax.set_xlim(1, 50)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(OUT_PNG), dpi=150)
        plt.close(fig)
        print(f"Saved plot -> {OUT_PNG}")
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()
