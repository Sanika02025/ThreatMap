"""
score_llm_sample.py — Scorer for RAG-ATT&CK Table 2 reproduction sample.

Reads ONLY:
  - artifacts/llm_cache/<exp_id>.jsonl (cached LLM responses)
  - data/tram2-data/multi_label.json (gold labels)
  - data/processed/mitre_attack_50_techniques.json (50-class KB)

Hard rules:
  - ZERO LLM calls or network access.
  - Hard-fail (raise AssertionError) if any expected sentence is missing from the cache.
  - Parse with regex r'\\b(T\\d{4}(?:\\.\\d{3})?)\\b', handle 'No Mapping' / 'None'.
  - Score strictly over the 50 classes.
  - Count and report unlisted predictions (excluded from precision denominator).
  - Compute micro P, R, F1 and macro F1 (only over classes with sample support).
  - Print comparison table next to paper targets.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_FILE = ROOT / "data" / "tram2-data" / "multi_label.json"
KB_FILE = ROOT / "data" / "processed" / "mitre_attack_50_techniques.json"
EMB_DIR = ROOT / "artifacts" / "embedding_cache"
CACHE_DIR = ROOT / "artifacts" / "llm_cache"
SAMPLE_RESULTS_FILE = ROOT / "artifacts" / "llm_sample_results.json"
REPORT_OUTPUT_FILE = ROOT / "artifacts" / "llm_sample_scored_report.json"

# Import builders from run_llm_sample to reproduce exact messages and hashes
from scripts.run_llm_sample import (
    MODEL,
    OPTIONS,
    RAG_K,
    SELECTED_DOC_INDICES,
    _cache_key,
    build_base_llm_messages,
    build_base_rag_messages,
    build_context1_messages,
    build_context_windows,
    build_optimized_rag_messages,
    build_sentence_index,
    get_top_k_techniques,
    load_kb,
    load_tram_data,
    select_sample,
)

# ── Paper Table 2 Targets (micro % and macro %) ──────────────────────────────
# From Schuurman (2025) Table 2:
# #19 BaseLLM (8B):       micro P=2.95%, R=12.46%, F1=4.77%, macro F1=4.27%
# #22 BaseRAG (8B, k=10): micro P=11.34%, R=55.08%, F1=18.81%, macro F1=19.47%
# #25 Optimized RAG (8B): micro P=8.77%, R=63.27%, F1=15.40%, macro F1=16.84%
# #27 Context 1 sent:     micro P=7.76%, R=54.31%, F1=13.58%, macro F1=15.60%
PAPER_TARGETS = {
    19: {"micro_p": 2.95, "micro_r": 12.46, "micro_f1": 4.77, "macro_f1": 4.27, "name": "BaseLLM 8B (#19)"},
    22: {"micro_p": 11.34, "micro_r": 55.08, "micro_f1": 18.81, "macro_f1": 19.47, "name": "BaseRAG 8B (#22)"},
    25: {"micro_p": 8.77, "micro_r": 63.27, "micro_f1": 15.40, "macro_f1": 16.84, "name": "Optimized RAG 8B (#25)"},
    27: {"micro_p": 7.76, "micro_r": 54.31, "micro_f1": 13.58, "macro_f1": 15.60, "name": "Context 1-sent (#27)"},
}

TECHNIQUE_REGEX = re.compile(r"\b(T\d{4}(?:\.\d{3})?)\b", re.IGNORECASE)


# ── Response Parser ──────────────────────────────────────────────────────────

def parse_llm_response(text: Optional[str], valid_50: Set[str]) -> Tuple[List[str], List[str]]:
    """
    Parses technique predictions from raw text response.
    Returns:
        (valid_50_predictions, unlisted_predictions)
    """
    if not text:
        return [], []

    cleaned = text.strip()
    if cleaned.lower() in ("no mapping", "no mapping.", "none", "none.", "n/a", ""):
        return [], []

    matches = TECHNIQUE_REGEX.findall(cleaned)
    valid_preds: List[str] = []
    unlisted_preds: List[str] = []
    seen = set()

    for m in matches:
        tid = m.upper().strip()
        if tid in seen:
            continue
        seen.add(tid)
        if tid in valid_50:
            valid_preds.append(tid)
        else:
            unlisted_preds.append(tid)

    return valid_preds, unlisted_preds


# ── Cache Loader ─────────────────────────────────────────────────────────────

def load_cache_file(exp_id: str) -> Dict[str, Dict]:
    cache_path = CACHE_DIR / f"{exp_id}.jsonl"
    if not cache_path.exists():
        return {}
    cache: Dict[str, Dict] = {}
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                cache[entry["hash"]] = entry
            except json.JSONDecodeError:
                pass
    return cache


# ── Scoring Logic ────────────────────────────────────────────────────────────

def score_predictions(
    predictions_per_sentence: List[List[str]],
    gold_per_sentence: List[List[str]],
    valid_50: Set[str],
) -> Dict[str, Any]:
    """
    Computes micro and macro metrics strictly over valid_50.
    Macro F1 is computed only over classes with support > 0 in gold_per_sentence.
    """
    assert len(predictions_per_sentence) == len(gold_per_sentence)

    # 1. Micro metrics
    total_tp = 0
    total_pred = 0
    total_gold = 0

    for preds, gold in zip(predictions_per_sentence, gold_per_sentence):
        p_set = set(preds)
        g_set = set(gold)
        tp = len(p_set & g_set)
        total_tp += tp
        total_pred += len(p_set)
        total_gold += len(g_set)

    micro_p = (total_tp / total_pred * 100.0) if total_pred > 0 else 0.0
    micro_r = (total_tp / total_gold * 100.0) if total_gold > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    # 2. Macro metrics (classes with support > 0 in sample)
    # Count gold occurrences per class
    gold_counts: Dict[str, int] = {c: 0 for c in valid_50}
    for gold in gold_per_sentence:
        for c in gold:
            if c in gold_counts:
                gold_counts[c] += 1

    supported_classes = [c for c, count in gold_counts.items() if count > 0]

    macro_f1_list = []
    class_details = {}

    for c in supported_classes:
        c_tp = 0
        c_fp = 0
        c_fn = 0
        for preds, gold in zip(predictions_per_sentence, gold_per_sentence):
            in_pred = c in preds
            in_gold = c in gold
            if in_pred and in_gold:
                c_tp += 1
            elif in_pred and not in_gold:
                c_fp += 1
            elif not in_pred and in_gold:
                c_fn += 1

        p_c = (c_tp / (c_tp + c_fp)) if (c_tp + c_fp) > 0 else 0.0
        r_c = (c_tp / (c_tp + c_fn)) if (c_tp + c_fn) > 0 else 0.0
        f1_c = (2 * p_c * r_c / (p_c + r_c)) if (p_c + r_c) > 0 else 0.0
        macro_f1_list.append(f1_c * 100.0)
        class_details[c] = {
            "support": gold_counts[c],
            "tp": c_tp,
            "fp": c_fp,
            "fn": c_fn,
            "f1": f1_c * 100.0,
        }

    macro_f1 = float(np.mean(macro_f1_list)) if macro_f1_list else 0.0

    return {
        "total_tp": total_tp,
        "total_pred": total_pred,
        "total_gold": total_gold,
        "micro_p": micro_p,
        "micro_r": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "num_supported_classes": len(supported_classes),
        "supported_classes": supported_classes,
        "class_details": class_details,
    }


def evaluate_experiment_from_cache(
    exp_num: int,
    exp_id: str,
    sentences_to_eval: List[Dict],
    techniques: List[Dict],
    valid_50: Set[str],
    tech_embs: Optional[np.ndarray],
    query_embs: Optional[np.ndarray],
    sent_to_labelled_idx: Dict[str, int],
    context_windows: Optional[Dict[int, Tuple[Optional[str], Optional[str]]]],
) -> Dict[str, Any]:
    """
    Verifies every expected sentence hash is in cache, extracts predictions,
    and computes metrics.
    RAISES AssertionError if ANY cache key is missing.
    """
    cache = load_cache_file(exp_id)
    if not cache:
        raise AssertionError(
            f"Cache file for {exp_id} is completely empty or missing! "
            f"Expected file at {CACHE_DIR / f'{exp_id}.jsonl'}"
        )

    all_preds_50: List[List[str]] = []
    all_unlisted_preds: List[List[str]] = []
    all_gold_50: List[List[str]] = []
    total_unlisted_count = 0

    for i, row in enumerate(sentences_to_eval):
        sentence = row["sentence"]

        # Re-derive expected messages to compute expected cache key
        if exp_num == 19:
            messages = build_base_llm_messages(sentence, techniques)
        elif exp_num in (22, 25, 27):
            lab_idx = sent_to_labelled_idx.get(sentence)
            if lab_idx is None:
                candidates: List[Dict] = []
            else:
                candidates = get_top_k_techniques(
                    lab_idx, tech_embs, query_embs, techniques, k=RAG_K
                )

            if exp_num == 22:
                messages = build_base_rag_messages(sentence, candidates)
            elif exp_num == 25:
                messages = build_optimized_rag_messages(sentence, candidates)
            elif exp_num == 27:
                prev_s, next_s = context_windows[i] if context_windows else (None, None)
                messages = build_context1_messages(sentence, candidates, prev_s, next_s)
        else:
            raise ValueError(f"Unknown experiment number: {exp_num}")

        expected_hash = _cache_key(MODEL, messages, OPTIONS)

        # HARD RULE: Fail if cache entry is missing
        if expected_hash not in cache:
            raise AssertionError(
                f"Scorer failure: Missing cached response for sentence index {i} "
                f"in experiment #{exp_num} ({exp_id}). Hash: {expected_hash}. "
                f"Sentence snippet: {repr(sentence[:60])}"
            )

        raw_resp = cache[expected_hash]["response"]
        preds_50, unlisted = parse_llm_response(raw_resp, valid_50)

        # Filter gold labels to 50 classes
        gold_raw = row.get("labels") or []
        gold_50 = [g for g in gold_raw if g in valid_50]

        all_preds_50.append(preds_50)
        all_unlisted_preds.append(unlisted)
        all_gold_50.append(gold_50)
        total_unlisted_count += len(unlisted)

    metrics = score_predictions(all_preds_50, all_gold_50, valid_50)
    metrics["total_unlisted"] = total_unlisted_count
    metrics["num_sentences"] = len(sentences_to_eval)
    metrics["num_gold_sentences"] = sum(1 for g in all_gold_50 if len(g) > 0)
    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Score RAG-ATT&CK Table 2 LLM sample runs strictly from disk cache.")
    parser.add_argument("--mode", choices=["pilot", "sample"], default="sample",
                        help="pilot=first 20 sentences; sample=full 2-report sample")
    parser.add_argument("--experiments", nargs="+", type=int, default=[19, 22, 25, 27],
                        help="Experiments to evaluate (19, 22, 25, 27)")
    args = parser.parse_args()

    print("=" * 70)
    print(" RAG-ATT&CK Table 2 Reproduction Scorer (Disk-Cache Only)")
    print("=" * 70)

    # 1. Load data & KB
    data = load_tram_data()
    sample, selected_titles = select_sample(data)
    techniques, valid_50 = load_kb()
    sent_to_labelled_idx = build_sentence_index(data)
    context_windows = build_context_windows(sample)

    eval_sentences = sample[:20] if args.mode == "pilot" else sample
    print(f"\nEvaluating mode: {args.mode.upper()}")
    print(f"Sample size: {len(eval_sentences)} sentences (out of {len(sample)} total in 2 selected reports)")
    print(f"Reports: {selected_titles[0]!r} and {selected_titles[1]!r}")

    # 2. Load embeddings if RAG experiments are included
    nomic_query_embs = None
    nomic_tech_embs = None
    attack_bert_query_embs = None
    attack_bert_tech_embs = None

    nomic_q_path = EMB_DIR / "nomic_queries_with_prefixes.npy"
    nomic_t_path = EMB_DIR / "nomic_tech_full_with_prefixes.npy"
    ab_q_path = EMB_DIR / "attack_bert_queries.npy"
    ab_t_path = EMB_DIR / "attack_bert_tech_full.npy"

    if any(e in args.experiments for e in [22, 27]):
        nomic_query_embs = np.load(str(nomic_q_path))
        nomic_tech_embs = np.load(str(nomic_t_path))

    if 25 in args.experiments:
        attack_bert_query_embs = np.load(str(ab_q_path))
        attack_bert_tech_embs = np.load(str(ab_t_path))

    exp_id_map = {
        19: "exp19_base_llm",
        22: "exp22_base_rag",
        25: "exp25_optimized_rag",
        27: "exp27_context1",
    }

    report_results: Dict[str, Any] = {
        "metadata": {
            "mode": args.mode,
            "num_sentences": len(eval_sentences),
            "reports": selected_titles,
            "experiments": args.experiments,
        },
        "experiments": {},
    }

    print("\n" + "-" * 70)
    print(f"{'Experiment':<25} | {'Micro P':<9} | {'Micro R':<9} | {'Micro F1':<9} | {'Macro F1':<9} | {'Unlisted':<8}")
    print("-" * 70)

    for exp_num in args.experiments:
        if exp_num not in exp_id_map:
            continue

        exp_id = exp_id_map[exp_num]

        if exp_num in (22, 27):
            q_embs = nomic_query_embs
            t_embs = nomic_tech_embs
        elif exp_num == 25:
            q_embs = attack_bert_query_embs
            t_embs = attack_bert_tech_embs
        else:
            q_embs = None
            t_embs = None

        try:
            metrics = evaluate_experiment_from_cache(
                exp_num=exp_num,
                exp_id=exp_id,
                sentences_to_eval=eval_sentences,
                techniques=techniques,
                valid_50=valid_50,
                tech_embs=t_embs,
                query_embs=q_embs,
                sent_to_labelled_idx=sent_to_labelled_idx,
                context_windows=context_windows,
            )
        except AssertionError as e:
            print(f"FAILED on #{exp_num} ({exp_id}): {e}")
            raise

        paper_t = PAPER_TARGETS[exp_num]
        name = paper_t["name"]

        print(
            f"{name:<25} | {metrics['micro_p']:>8.2f}% | {metrics['micro_r']:>8.2f}% | "
            f"{metrics['micro_f1']:>8.2f}% | {metrics['macro_f1']:>8.2f}% | {metrics['total_unlisted']:>8d}"
        )
        print(
            f"{'  (Paper Target)':<25} | {paper_t['micro_p']:>8.2f}% | {paper_t['micro_r']:>8.2f}% | "
            f"{paper_t['micro_f1']:>8.2f}% | {paper_t['macro_f1']:>8.2f}% | {'-':>8}"
        )
        print("-" * 70)

        report_results["experiments"][f"exp_{exp_num}"] = {
            "name": name,
            "exp_id": exp_id,
            "metrics": metrics,
            "paper_target": paper_t,
        }

    # Save scored report to JSON
    with open(REPORT_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report_results, f, indent=2, ensure_ascii=False)
    print(f"\nScored metrics saved to {REPORT_OUTPUT_FILE}")

    print("\n[NOTE] Smoke test comparison disclaimer:")
    print("  This evaluation ran on a ~150-sentence sample (2 whole reports) as a smoke test.")
    print("  Compare direction and relative ordering, not exact absolute numbers:")
    print("  - BaseLLM (#19) is expected to have the lowest recall and F1.")
    print("  - RAG runs (#22, #25, #27) substantially elevate recall over BaseLLM.")


if __name__ == "__main__":
    main()
