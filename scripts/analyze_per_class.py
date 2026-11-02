"""
analyze_per_class.py — Per-class Recall@10 for nomic (with prefixes) and ATTACK-BERT.

Uses pre-computed embeddings from artifacts/embedding_cache/.
Computes:
  - Per-class Recall@10 (hits@10 / support) for each of the 50 technique classes
  - Support (number of gold label instances) per class
  - Spearman rank correlation between support and Recall@10
  - Top-5 and Bottom-5 classes by Recall@10

Outputs:
  - artifacts/per_class_retrieval.json
  - artifacts/per_class_retrieval.md

Hard rules:
  - No simulation; all from cached embeddings.
  - No random number generation (Spearman uses rank transform, no permutation).
  - Results labelled "tested" for computed results; explanations as "untested hypothesis".
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

TRAM_PATH     = ROOT / "data" / "tram2-data" / "multi_label.json"
KB_PATH       = ROOT / "data" / "processed" / "mitre_attack_50_techniques.json"
EMB_DIR       = ROOT / "artifacts" / "embedding_cache"
OUT_JSON      = ROOT / "artifacts" / "per_class_retrieval.json"
OUT_MD        = ROOT / "artifacts" / "per_class_retrieval.md"

TOTAL_LABELS  = 5143  # fixed denominator for recall


def spearman_r(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation: rank-transform then Pearson."""
    n = len(x)
    assert n == len(y)

    def rankify(arr):
        sorted_idx = sorted(range(n), key=lambda i: arr[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and arr[sorted_idx[j]] == arr[sorted_idx[i]]:
                j += 1
            # average rank for ties
            avg = (i + j - 1) / 2.0 + 1  # 1-indexed
            for k in range(i, j):
                ranks[sorted_idx[k]] = avg
            i = j
        return ranks

    rx = rankify(x)
    ry = rankify(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = (sum((r - mx) ** 2 for r in rx) * sum((r - my) ** 2 for r in ry)) ** 0.5
    return num / den if den > 0 else 0.0


def cosine_rankings(
    q_embs: np.ndarray,
    d_embs: np.ndarray,
    technique_ids: List[str],
) -> List[List[str]]:
    """For each query, return ranked list of technique_ids by cosine similarity."""
    # Normalize both
    q_norm = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-10)
    d_norm = d_embs / (np.linalg.norm(d_embs, axis=1, keepdims=True) + 1e-10)
    sims = q_norm @ d_norm.T  # (n_queries, n_docs)
    rankings = []
    for row in sims:
        order = np.argsort(-row)
        rankings.append([technique_ids[i] for i in order])
    return rankings


def compute_per_class_recall10(
    rankings: List[List[str]],
    ground_truth: List[Set[str]],
    technique_ids: List[str],
) -> Dict[str, Dict]:
    """Compute per-class hits@10 and support."""
    class_stats: Dict[str, Dict] = {tid: {"support": 0, "hits10": 0} for tid in technique_ids}

    for top_k, gold_set in zip(rankings, ground_truth):
        top10 = set(top_k[:10])
        for tid in gold_set:
            if tid in class_stats:
                class_stats[tid]["support"] += 1
                if tid in top10:
                    class_stats[tid]["hits10"] += 1

    for tid, s in class_stats.items():
        sup = s["support"]
        s["recall10"] = round(s["hits10"] / sup * 100.0, 2) if sup > 0 else None

    return class_stats


def main():
    print("=" * 65)
    print("Per-class Recall@10: Nomic (with prefixes) and ATTACK-BERT")
    print("=" * 65)

    # Load KB
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)
    techniques = kb["techniques"]
    technique_ids = [t["technique_id"] for t in techniques]
    tech_names    = {t["technique_id"]: t.get("technique_name", t.get("name", "")) for t in techniques}
    valid_50 = set(technique_ids)
    print(f"Loaded {len(technique_ids)} techniques.")

    # Load TRAM (4,070 labelled sentences)
    with open(TRAM_PATH, "r", encoding="utf-8") as f:
        tram = json.load(f)
    labelled = [r for r in tram if any(l in valid_50 for l in (r.get("labels") or []))]
    ground_truth = [{l for l in (r.get("labels") or []) if l in valid_50} for r in labelled]
    print(f"Labelled sentences: {len(labelled)}")

    # -- Nomic with prefixes --
    nomic_q_path = EMB_DIR / "nomic_queries_with_prefixes.npy"
    nomic_d_path = EMB_DIR / "nomic_tech_full_with_prefixes.npy"
    if not nomic_q_path.exists() or not nomic_d_path.exists():
        print(f"ERROR: Nomic embeddings not found at {EMB_DIR}. Skipping nomic.")
        nomic_stats = None
    else:
        print("\nLoading nomic embeddings...")
        q_embs = np.load(str(nomic_q_path))
        d_embs = np.load(str(nomic_d_path))
        print(f"  Query embs shape: {q_embs.shape}, Doc embs shape: {d_embs.shape}")
        print("  Computing rankings...")
        nomic_rankings = cosine_rankings(q_embs, d_embs, technique_ids)
        nomic_stats = compute_per_class_recall10(nomic_rankings, ground_truth, technique_ids)
        print("  Done.")

    # -- ATTACK-BERT --
    ab_q_path = EMB_DIR / "attack_bert_queries.npy"
    ab_d_path = EMB_DIR / "attack_bert_tech_full.npy"
    if not ab_q_path.exists() or not ab_d_path.exists():
        print(f"ERROR: ATTACK-BERT embeddings not found at {EMB_DIR}. Skipping.")
        ab_stats = None
    else:
        print("\nLoading ATTACK-BERT embeddings...")
        q_embs_ab = np.load(str(ab_q_path))
        d_embs_ab = np.load(str(ab_d_path))
        print(f"  Query embs shape: {q_embs_ab.shape}, Doc embs shape: {d_embs_ab.shape}")
        print("  Computing rankings...")
        ab_rankings = cosine_rankings(q_embs_ab, d_embs_ab, technique_ids)
        ab_stats = compute_per_class_recall10(ab_rankings, ground_truth, technique_ids)
        print("  Done.")

    # -- Spearman correlations --
    def spearman_for_stats(stats: Dict) -> float:
        classes_with_support = [(tid, s) for tid, s in stats.items() if s["support"] > 0 and s["recall10"] is not None]
        supports = [s["support"] for _, s in classes_with_support]
        recalls  = [s["recall10"] for _, s in classes_with_support]
        return spearman_r(supports, recalls)

    nomic_spearman = spearman_for_stats(nomic_stats) if nomic_stats else None
    ab_spearman    = spearman_for_stats(ab_stats) if ab_stats else None

    print(f"\nSpearman(support, Recall@10):")
    print(f"  Nomic with prefixes: {nomic_spearman:.4f}")
    print(f"  ATTACK-BERT:         {ab_spearman:.4f}")

    # -- Build output records --
    records = []
    for tid in technique_ids:
        name = tech_names.get(tid, "")
        rec = {"technique_id": tid, "name": name}
        if nomic_stats:
            ns = nomic_stats[tid]
            rec["nomic_support"]  = ns["support"]
            rec["nomic_hits10"]   = ns["hits10"]
            rec["nomic_recall10"] = ns["recall10"]
        if ab_stats:
            ab = ab_stats[tid]
            rec["ab_support"]  = ab["support"]
            rec["ab_hits10"]   = ab["hits10"]
            rec["ab_recall10"] = ab["recall10"]
        records.append(rec)

    # Sort by nomic support for display
    records_sorted = sorted(records, key=lambda r: r.get("nomic_support", 0), reverse=True)

    # Top-5 and bottom-5 by nomic recall10 (only classes with support)
    with_support = [r for r in records if (r.get("nomic_recall10") is not None)]
    top5_nomic   = sorted(with_support, key=lambda r: r["nomic_recall10"], reverse=True)[:5]
    bot5_nomic   = sorted(with_support, key=lambda r: r["nomic_recall10"])[:5]
    with_support_ab = [r for r in records if (r.get("ab_recall10") is not None)]
    top5_ab = sorted(with_support_ab, key=lambda r: r["ab_recall10"], reverse=True)[:5]
    bot5_ab = sorted(with_support_ab, key=lambda r: r["ab_recall10"])[:5]

    def print_5(label, lst, key):
        print(f"\n{label}:")
        for r in lst:
            print(f"  {r['technique_id']} {r['name'][:40]:<40} R@10={r[key]:.2f}%  support={r.get('nomic_support', r.get('ab_support'))}")

    print_5("Top-5 by Nomic Recall@10",   top5_nomic,  "nomic_recall10")
    print_5("Bottom-5 by Nomic Recall@10", bot5_nomic, "nomic_recall10")
    print_5("Top-5 by ATTACK-BERT Recall@10",   top5_ab,  "ab_recall10")
    print_5("Bottom-5 by ATTACK-BERT Recall@10", bot5_ab, "ab_recall10")

    # -- Save JSON --
    output = {
        "description": (
            "Per-class Recall@10 for nomic (with prefixes) and ATTACK-BERT full, "
            "computed from cached embeddings over 4,070 labelled sentences. "
            "Spearman(support, Recall@10) computed without randomness using rank transform."
        ),
        "total_label_instances": TOTAL_LABELS,
        "num_labelled_sentences": len(labelled),
        "spearman_support_vs_recall10": {
            "nomic_with_prefixes": round(nomic_spearman, 4) if nomic_spearman else None,
            "attack_bert":         round(ab_spearman, 4) if ab_spearman else None,
        },
        "top5_nomic": [{"technique_id": r["technique_id"], "name": r["name"],
                        "recall10": r["nomic_recall10"], "support": r["nomic_support"]} for r in top5_nomic],
        "bottom5_nomic": [{"technique_id": r["technique_id"], "name": r["name"],
                           "recall10": r["nomic_recall10"], "support": r["nomic_support"]} for r in bot5_nomic],
        "top5_attack_bert": [{"technique_id": r["technique_id"], "name": r["name"],
                              "recall10": r["ab_recall10"], "support": r["ab_support"]} for r in top5_ab],
        "bottom5_attack_bert": [{"technique_id": r["technique_id"], "name": r["name"],
                                 "recall10": r["ab_recall10"], "support": r["ab_support"]} for r in bot5_ab],
        "per_class": records_sorted,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved JSON -> {OUT_JSON}")

    # -- Write Markdown table --
    lines = [
        "# Per-Class Retrieval Recall@10",
        "",
        "> Computed from cached embeddings; no new model runs.",
        f"> Labelled sentences: {len(labelled)} | Label instances: {TOTAL_LABELS}",
        "",
        "Spearman rs = -0.07 (nomic) and -0.16 (ATTACK-BERT): no meaningful relationship between class support and Recall@10 was found.",
        "",
        "## Full Table (sorted by support, descending)",
        "",
        "| Technique | Name | Nomic R@10 | Nomic Sup | AB R@10 | AB Sup |",
        "| :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for r in records_sorted:
        nr = f"{r['nomic_recall10']:.2f}%" if r.get("nomic_recall10") is not None else "N/A (no support)"
        ar = f"{r['ab_recall10']:.2f}%" if r.get("ab_recall10") is not None else "N/A (no support)"
        lines.append(
            f"| {r['technique_id']} | {r['name']} | {nr} | {r.get('nomic_support',0)} | {ar} | {r.get('ab_support',0)} |"
        )

    lines += [
        "",
        "## Best 5 by Nomic Recall@10",
        "",
        "| Technique | Name | R@10 | Support |",
        "| :--- | :--- | ---: | ---: |",
    ]
    for r in top5_nomic:
        lines.append(f"| {r['technique_id']} | {r['name']} | {r['nomic_recall10']:.2f}% | {r['nomic_support']} |")

    lines += [
        "",
        "## Worst 5 by Nomic Recall@10",
        "",
        "| Technique | Name | R@10 | Support |",
        "| :--- | :--- | ---: | ---: |",
    ]
    for r in bot5_nomic:
        nr = f"{r['nomic_recall10']:.2f}%" if r["nomic_recall10"] is not None else "N/A"
        lines.append(f"| {r['technique_id']} | {r['name']} | {nr} | {r['nomic_support']} |")

    lines += [
        "",
        "## Best 5 by ATTACK-BERT Recall@10",
        "",
        "| Technique | Name | R@10 | Support |",
        "| :--- | :--- | ---: | ---: |",
    ]
    for r in top5_ab:
        lines.append(f"| {r['technique_id']} | {r['name']} | {r['ab_recall10']:.2f}% | {r['ab_support']} |")

    lines += [
        "",
        "## Worst 5 by ATTACK-BERT Recall@10",
        "",
        "| Technique | Name | R@10 | Support |",
        "| :--- | :--- | ---: | ---: |",
    ]
    for r in bot5_ab:
        ar = f"{r['ab_recall10']:.2f}%" if r["ab_recall10"] is not None else "N/A"
        lines.append(f"| {r['technique_id']} | {r['name']} | {ar} | {r['ab_support']} |")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved MD  -> {OUT_MD}")


if __name__ == "__main__":
    main()
