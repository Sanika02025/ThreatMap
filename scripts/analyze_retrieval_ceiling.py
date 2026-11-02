"""
analyze_retrieval_ceiling.py — Ceiling analysis for RAG experiments #22, #25, #27.

For each of the 191 sample sentences, reads the LLM cache to extract:
  (a) number of gold in-scope labels (54)
  (b) how many gold labels appear among the 10 candidates shown in that prompt (candidate recall)
  (c) how many of those the LLM actually predicted (selection rate = TP / gold-in-candidates)
  (d) gold labels NOT in the candidates (retrieval misses)
  (e) gold labels in the candidates but NOT predicted (LLM misses)
State which stage loses more.

Outputs: artifacts/retrieval_ceiling.json
Hard rules:
  - No simulation; all data from real cache files.
  - No random number generation.
  - All explanations of gaps labelled "tested" or "untested hypothesis".
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_llm_sample import (
    MODEL,
    OPTIONS,
    RAG_K,
    _cache_key,
    build_base_rag_messages,
    build_context1_messages,
    build_context_windows,
    build_optimized_rag_messages,
    get_top_k_techniques,
    load_kb,
    load_tram_data,
    select_sample,
)
from scripts.score_llm_sample import parse_llm_response, load_cache_file

CACHE_DIR  = ROOT / "artifacts" / "llm_cache"
EMB_DIR    = ROOT / "artifacts" / "embedding_cache"
OUT_FILE   = ROOT / "artifacts" / "retrieval_ceiling.json"


def analyze_experiment(
    exp_num: int,
    exp_id: str,
    sample: List[Dict],
    techniques: List[Dict],
    valid_50: Set[str],
    tech_embs,
    query_embs,
    context_windows,
) -> Dict:
    cache = load_cache_file(exp_id)
    if not cache:
        raise FileNotFoundError(f"Cache missing for {exp_id}")

    total_gold = 0
    total_gold_in_candidates = 0
    total_gold_not_in_candidates = 0
    total_tp = 0
    total_gold_missed_by_llm = 0
    per_sentence = []

    for i, row in enumerate(sample):
        sentence = row["sentence"]
        gold_raw = row.get("labels") or []
        gold_50 = {g for g in gold_raw if g in valid_50}

        # Derive candidates exactly as done in run_llm_sample
        candidates = get_top_k_techniques(i, tech_embs, query_embs, techniques, k=RAG_K)
        cand_ids = {c["technique_id"] for c in candidates}

        if exp_num == 22:
            messages = build_base_rag_messages(sentence, candidates)
        elif exp_num == 25:
            messages = build_optimized_rag_messages(sentence, candidates)
        elif exp_num == 27:
            prev_s, next_s = context_windows[i] if context_windows else (None, None)
            messages = build_context1_messages(sentence, candidates, prev_s, next_s)
        else:
            raise ValueError(f"Unknown exp_num: {exp_num}")

        h = _cache_key(MODEL, messages, OPTIONS)
        if h not in cache:
            raise AssertionError(f"Missing cache key {h} for sent {i} in {exp_id}")

        raw_resp = cache[h]["response"]
        preds_50, unlisted = parse_llm_response(raw_resp, valid_50)
        preds_set = set(preds_50)

        gold_in_cands = gold_50 & cand_ids
        gold_not_in_cands = gold_50 - cand_ids
        tp = gold_in_cands & preds_set
        llm_miss = gold_in_cands - preds_set

        total_gold += len(gold_50)
        total_gold_in_candidates += len(gold_in_cands)
        total_gold_not_in_candidates += len(gold_not_in_cands)
        total_tp += len(tp)
        total_gold_missed_by_llm += len(llm_miss)

        if gold_50:
            per_sentence.append({
                "sentence_idx": i,
                "sentence": sentence[:80],
                "gold": sorted(gold_50),
                "candidates": sorted(cand_ids),
                "gold_in_candidates": sorted(gold_in_cands),
                "gold_not_in_candidates": sorted(gold_not_in_cands),
                "predicted": sorted(preds_set),
                "tp": sorted(tp),
                "llm_miss": sorted(llm_miss),
            })

    cand_recall = round(total_gold_in_candidates / total_gold * 100, 2) if total_gold > 0 else 0.0
    selection_rate = round(total_tp / total_gold_in_candidates * 100, 2) if total_gold_in_candidates > 0 else 0.0

    larger_loss = (
        "retrieval stage" if total_gold_not_in_candidates > total_gold_missed_by_llm
        else "LLM selection stage" if total_gold_missed_by_llm > total_gold_not_in_candidates
        else "equal loss"
    )

    return {
        "exp_id": exp_id,
        "total_gold_labels": total_gold,
        "gold_in_candidates": total_gold_in_candidates,
        "candidate_recall_pct": cand_recall,
        "gold_not_in_candidates": total_gold_not_in_candidates,
        "tp": total_tp,
        "gold_missed_by_llm": total_gold_missed_by_llm,
        "selection_rate_pct": selection_rate,
        "larger_loss_stage": larger_loss,
        "per_sentence": per_sentence,
    }


def main():
    print("=" * 65)
    print("Retrieval Ceiling Analysis — #22, #25, #27 (191-sentence sample)")
    print("Computed strictly from artifacts/llm_cache/ and cached embeddings")
    print("=" * 65)

    data = load_tram_data()
    sample, selected_titles = select_sample(data)
    techniques, valid_50 = load_kb()
    context_windows = build_context_windows(sample)

    print(f"Sample size: {len(sample)} sentences from {len(selected_titles)} reports")

    # Load embeddings
    nomic_q = np.load(str(EMB_DIR / "sample_191_nomic_queries.npy"))
    nomic_t = np.load(str(EMB_DIR / "nomic_tech_full_with_prefixes.npy"))
    ab_q = np.load(str(EMB_DIR / "sample_191_attack_bert_queries.npy"))
    ab_t = np.load(str(EMB_DIR / "attack_bert_tech_full.npy"))

    configs = [
        (22, "exp22_base_rag",      "#22 BaseRAG 8B",        nomic_t, nomic_q),
        (25, "exp25_optimized_rag", "#25 Optimized RAG 8B",  ab_t,    ab_q),
        (27, "exp27_context1",      "#27 Context 1-sent",    ab_t,    ab_q),
    ]

    results = {}
    for exp_num, exp_id, label, t_embs, q_embs in configs:
        res = analyze_experiment(
            exp_num=exp_num,
            exp_id=exp_id,
            sample=sample,
            techniques=techniques,
            valid_50=valid_50,
            tech_embs=t_embs,
            query_embs=q_embs,
            context_windows=context_windows,
        )
        res_summary = {k: v for k, v in res.items() if k != "per_sentence"}
        res_summary["label"] = label
        results[exp_id] = res_summary

        print(f"\n{label} ({exp_id}):")
        print(f"  (a) Gold in-scope labels:                {res['total_gold_labels']}")
        print(f"  (b) Gold in top-10 candidates:           {res['gold_in_candidates']} (candidate recall: {res['candidate_recall_pct']}%)")
        print(f"  (c) Predicted by LLM (TP):               {res['tp']} (selection rate: {res['selection_rate_pct']}%)")
        print(f"  (d) Retrieval misses (not in candidates): {res['gold_not_in_candidates']}")
        print(f"  (e) LLM misses (in cands, not predicted): {res['gold_missed_by_llm']}")
        print(f"  Stage losing more:                        {res['larger_loss_stage']} (tested)")

    print("\n" + "=" * 80)
    print("SHORT TABLE: RETRIEVAL CEILING & ERROR DECOMPOSITION")
    print("=" * 80)
    print(f"{'Experiment':<24} | {'Gold':>4} | {'In Cand':>7} | {'Cand R@10':>9} | {'TP':>4} | {'Retr Miss':>9} | {'LLM Miss':>8} | {'Sel Rate':>8} | {'Larger Loss':<18}")
    print("-" * 80)
    for exp_id, r in results.items():
        print(f"{r['label']:<24} | {r['total_gold_labels']:>4} | {r['gold_in_candidates']:>7} | {r['candidate_recall_pct']:>8.2f}% | {r['tp']:>4} | {r['gold_not_in_candidates']:>9} | {r['gold_missed_by_llm']:>8} | {r['selection_rate_pct']:>7.2f}% | {r['larger_loss_stage']:<18}")
    print("=" * 80)
    print("Definitions:")
    print("  Gold: total in-scope gold labels in the 191-sentence sample (54).")
    print("  In Cand: gold labels appearing in top-10 candidates presented to LLM.")
    print("  Cand R@10: candidate recall = In Cand / Gold.")
    print("  TP: gold labels correctly predicted by the LLM.")
    print("  Retr Miss: gold labels NOT retrieved into top-10 candidates (Gold - In Cand).")
    print("  LLM Miss: gold labels present in candidates that LLM failed to select (In Cand - TP).")
    print("  Sel Rate: LLM selection rate = TP / In Cand.")
    print("  Larger Loss: comparison of Retr Miss vs LLM Miss.")

    output_data = {
        "description": "Retrieval ceiling and error decomposition for #22, #25, #27 on 191-sentence sample.",
        "sample_sentences": len(sample),
        "total_gold_in_scope": 54,
        "results": results,
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved results to {OUT_FILE}")


if __name__ == "__main__":
    main()
