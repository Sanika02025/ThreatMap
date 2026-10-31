"""
score_llm_sample.py — Scorer for RAG-ATT&CK Table 2 reproduction sample.

Reads ONLY:
  - artifacts/llm_cache/<exp_id>.jsonl (cached LLM responses)
  - data/tram2-data/multi_label.json (gold labels)
  - data/processed/mitre_attack_50_techniques.json (50-class KB)

Hard rules:
  - ZERO LLM calls or network access.
  - Hard-fail (raise AssertionError) if any expected sentence is missing from cache.
  - Parse with regex r'\\b(T\\d{4}(?:\\.\\d{3})?)\\b', handle 'No Mapping' / 'None'.
  - Score strictly over the 50 classes.
  - Count and report unlisted predictions (excluded from precision denominator).
  - Compute micro P, R, F1 and macro F1 (only over classes with sample support).
  - Bootstrap 95% CI over sentences (seed=42; this is the ONLY randomness in the scorer,
    documented here: 1000 bootstrap resamples of sentence-level outcomes, seed=42).
  - Print per-experiment: TP/FP/FN counts, micro P/R/F1, unlisted count,
    num sentences and in-scope labels, macro F1 with class coverage (flagged unreliable).
  - Print comparison table next to paper targets.
  - State clearly this is a ~191-sentence sample, not the paper's 19,178.
"""

import argparse
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
# From Schuurman (2025) Table 2 (exact values):
# #19 BaseLLM 8B:       micro P/R/F1 = 2.95/12.46/4.77  | macro P/R/F1 = 4.30/14.07/4.27
# #22 BaseRAG 8B:       micro P/R/F1 = 11.34/55.08/18.81 | macro P/R/F1 = 13.56/58.69/19.47
# #25 Optimized k=10:   micro P/R/F1 = 8.77/63.27/15.40  | macro P/R/F1 = 10.65/65.98/16.84
# #27 Context 1 sent:   micro P/R/F1 = 7.76/54.31/13.58  | macro P/R/F1 = 10.21/57.59/15.60
PAPER_TARGETS = {
    19: {
        "name": "BaseLLM 8B (#19)",
        "micro_p": 2.95, "micro_r": 12.46, "micro_f1": 4.77,
        "macro_p": 4.30, "macro_r": 14.07, "macro_f1": 4.27,
    },
    22: {
        "name": "BaseRAG 8B (#22)",
        "micro_p": 11.34, "micro_r": 55.08, "micro_f1": 18.81,
        "macro_p": 13.56, "macro_r": 58.69, "macro_f1": 19.47,
    },
    25: {
        "name": "Optimized RAG 8B (#25)",
        "micro_p": 8.77, "micro_r": 63.27, "micro_f1": 15.40,
        "macro_p": 10.65, "macro_r": 65.98, "macro_f1": 16.84,
    },
    27: {
        "name": "Context 1-sent (#27)",
        "micro_p": 7.76, "micro_r": 54.31, "micro_f1": 13.58,
        "macro_p": 10.21, "macro_r": 57.59, "macro_f1": 15.60,
    },
}

TECHNIQUE_REGEX = re.compile(r"\b(T\d{4}(?:\.\d{3})?)\b", re.IGNORECASE)

# Bootstrap parameters — ONLY randomness in scorer, documented.
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


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

def _micro_f1_from_pairs(preds_per_sent: List[List[str]], gold_per_sent: List[List[str]]) -> float:
    """Compute micro F1 from per-sentence prediction/gold lists."""
    total_tp = total_pred = total_gold = 0
    for preds, gold in zip(preds_per_sent, gold_per_sent):
        p_set = set(preds)
        g_set = set(gold)
        total_tp += len(p_set & g_set)
        total_pred += len(p_set)
        total_gold += len(g_set)
    micro_p = (total_tp / total_pred * 100.0) if total_pred > 0 else 0.0
    micro_r = (total_tp / total_gold * 100.0) if total_gold > 0 else 0.0
    return (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0


def bootstrap_micro_f1_ci(
    preds_per_sent: List[List[str]],
    gold_per_sent: List[List[str]],
    n: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float]:
    """
    Bootstrap 95% CI for micro F1 over sentences.

    RANDOMNESS NOTE: This function uses numpy.random.default_rng(seed=42) to
    resample sentences with replacement. This is the ONLY source of randomness
    in the scorer. It is seeded for reproducibility (seed=42) and used solely
    for confidence interval estimation, not for any evaluation decision.
    """
    rng = np.random.default_rng(seed)
    N = len(preds_per_sent)
    f1_samples = []
    for _ in range(n):
        idxs = rng.integers(0, N, size=N)
        boot_preds = [preds_per_sent[i] for i in idxs]
        boot_gold = [gold_per_sent[i] for i in idxs]
        f1_samples.append(_micro_f1_from_pairs(boot_preds, boot_gold))
    lo = float(np.percentile(f1_samples, 2.5))
    hi = float(np.percentile(f1_samples, 97.5))
    return lo, hi


def score_predictions(
    predictions_per_sentence: List[List[str]],
    gold_per_sentence: List[List[str]],
    valid_50: Set[str],
) -> Dict[str, Any]:
    """
    Computes micro and macro metrics strictly over valid_50.
    Macro F1 is computed only over classes with support > 0 in gold_per_sentence.
    Includes TP/FP/FN counts and bootstrap 95% CI for micro F1.
    """
    assert len(predictions_per_sentence) == len(gold_per_sentence)

    # 1. Micro metrics with TP/FP/FN
    total_tp = total_fp = total_fn = 0

    for preds, gold in zip(predictions_per_sentence, gold_per_sentence):
        p_set = set(preds)
        g_set = set(gold)
        tp = len(p_set & g_set)
        fp = len(p_set - g_set)
        fn = len(g_set - p_set)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    total_pred = total_tp + total_fp
    total_gold = total_tp + total_fn

    micro_p = (total_tp / total_pred * 100.0) if total_pred > 0 else 0.0
    micro_r = (total_tp / total_gold * 100.0) if total_gold > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    # Bootstrap 95% CI (seed=42, documented above)
    ci_lo, ci_hi = bootstrap_micro_f1_ci(predictions_per_sentence, gold_per_sentence)

    # 2. Macro metrics (only over classes with support > 0 in sample)
    gold_counts: Dict[str, int] = {c: 0 for c in valid_50}
    for gold in gold_per_sentence:
        for c in gold:
            if c in gold_counts:
                gold_counts[c] += 1

    supported_classes = [c for c, count in gold_counts.items() if count > 0]

    macro_p_list: List[float] = []
    macro_r_list: List[float] = []
    macro_f1_list: List[float] = []
    class_details: Dict[str, Any] = {}

    for c in supported_classes:
        c_tp = c_fp = c_fn = 0
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
        macro_p_list.append(p_c * 100.0)
        macro_r_list.append(r_c * 100.0)
        macro_f1_list.append(f1_c * 100.0)
        class_details[c] = {
            "support": gold_counts[c],
            "tp": c_tp,
            "fp": c_fp,
            "fn": c_fn,
            "precision": p_c * 100.0,
            "recall": r_c * 100.0,
            "f1": f1_c * 100.0,
        }

    macro_p = float(np.mean(macro_p_list)) if macro_p_list else 0.0
    macro_r = float(np.mean(macro_r_list)) if macro_r_list else 0.0
    macro_f1 = float(np.mean(macro_f1_list)) if macro_f1_list else 0.0

    return {
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_pred": total_pred,
        "total_gold": total_gold,
        "micro_p": micro_p,
        "micro_r": micro_r,
        "micro_f1": micro_f1,
        "micro_f1_ci_lo": ci_lo,
        "micro_f1_ci_hi": ci_hi,
        "macro_p": macro_p,
        "macro_r": macro_r,
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
    empty_candidates_count = 0

    for i, row in enumerate(sentences_to_eval):
        sentence = row["sentence"]

        # Re-derive expected messages to compute expected cache key
        if exp_num == 19:
            messages = build_base_llm_messages(sentence, techniques)
        elif exp_num in (22, 25, 27):
            assert query_embs is not None and tech_embs is not None, "Missing embeddings for RAG experiment"
            candidates = get_top_k_techniques(i, tech_embs, query_embs, techniques, k=RAG_K)
            if len(candidates) == 0:
                empty_candidates_count += 1
            assert len(candidates) == RAG_K, f"Expected {RAG_K} candidates, got {len(candidates)}"

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
    metrics["empty_candidate_prompts"] = empty_candidates_count
    metrics["num_sentences"] = len(sentences_to_eval)
    metrics["num_sentences_with_gold"] = sum(1 for g in all_gold_50 if len(g) > 0)
    metrics["num_in_scope_labels"] = sum(len(g) for g in all_gold_50)
    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────

def print_experiment_block(exp_num: int, name: str, metrics: Dict[str, Any], paper_t: Dict[str, Any]):
    """Print a detailed block for one experiment."""
    n_sent = metrics["num_sentences"]
    n_gold_sent = metrics["num_sentences_with_gold"]
    n_labels = metrics["num_in_scope_labels"]
    n_classes = metrics["num_supported_classes"]
    tp = metrics["total_tp"]
    fp = metrics["total_fp"]
    fn = metrics["total_fn"]
    unlisted = metrics["total_unlisted"]

    print(f"\n{'=' * 70}")
    print(f"  #{exp_num}: {name}")
    print(f"{'=' * 70}")
    print(f"  Sentences evaluated:      {n_sent}")
    print(f"  Sentences with gold label:{n_gold_sent}")
    print(f"  In-scope gold labels:     {n_labels}  (over {n_classes} classes with support)")
    print(f"  Unlisted predictions:     {unlisted}  (excluded from precision)")
    if exp_num in (22, 25, 27):
        print(f"  Empty candidate prompts:  {metrics.get('empty_candidate_prompts', 0)}/{n_sent}")
    print()
    print(f"  Counts:   TP={tp}  FP={fp}  FN={fn}")
    print()

    mp, mr, mf = metrics["micro_p"], metrics["micro_r"], metrics["micro_f1"]
    ci_lo, ci_hi = metrics["micro_f1_ci_lo"], metrics["micro_f1_ci_hi"]
    print(f"  Micro P : {mp:6.2f}%   (Paper: {paper_t['micro_p']:.2f}%)")
    print(f"  Micro R : {mr:6.2f}%   (Paper: {paper_t['micro_r']:.2f}%)")
    print(f"  Micro F1: {mf:6.2f}%   (Paper: {paper_t['micro_f1']:.2f}%)")
    print(f"  95% CI for Micro F1 (bootstrap, N={BOOTSTRAP_N}, seed={BOOTSTRAP_SEED}): "
          f"[{ci_lo:.2f}%, {ci_hi:.2f}%]")
    print()

    mac_p, mac_r, mac_f = metrics["macro_p"], metrics["macro_r"], metrics["macro_f1"]
    print(f"  Macro P : {mac_p:6.2f}%   (Paper: {paper_t['macro_p']:.2f}%)")
    print(f"  Macro R : {mac_r:6.2f}%   (Paper: {paper_t['macro_r']:.2f}%)")
    print(f"  Macro F1: {mac_f:6.2f}%   (Paper: {paper_t['macro_f1']:.2f}%)")
    print(f"  [UNRELIABLE: macro averaged over only {n_classes}/50 classes with support; "
          f"tiny sample (N={n_sent} sentences)]")


def main():
    parser = argparse.ArgumentParser(
        description="Score RAG-ATT&CK Table 2 LLM sample runs strictly from disk cache."
    )
    parser.add_argument("--mode", choices=["pilot", "sample"], default="sample",
                        help="pilot=first 20 sentences; sample=full 2-report sample")
    parser.add_argument("--experiments", nargs="+", type=int, default=[19, 22, 25, 27],
                        help="Experiments to evaluate (19, 22, 25, 27)")
    args = parser.parse_args()

    print("=" * 70)
    print(" RAG-ATT&CK Table 2 Reproduction Scorer (Disk-Cache Only)")
    print("=" * 70)
    print()
    print("!!! IMPORTANT DISCLAIMER !!!")
    print("  This scorer evaluates a ~191-sentence sample (2 complete CTI reports).")
    print("  The paper (Schuurman 2025) evaluated on 19,178 sentences (the full TRAM corpus).")
    print("  Metrics from this sample CANNOT be directly compared to paper numbers.")
    print("  Compare direction and relative ordering only — not exact values.")
    print()
    print(f"  Bootstrap CI: {BOOTSTRAP_N} resamples, seed={BOOTSTRAP_SEED}.")
    print(f"  This is the ONLY randomness in the scorer (documented in bootstrap_micro_f1_ci).")

    # 1. Load data & KB
    data = load_tram_data()
    sample, selected_titles = select_sample(data)
    techniques, valid_50 = load_kb()
    sent_to_labelled_idx = build_sentence_index(data)
    context_windows = build_context_windows(sample)

    eval_sentences = sample[:20] if args.mode == "pilot" else sample
    print(f"\nEvaluating mode: {args.mode.upper()}")
    print(f"Sample size: {len(eval_sentences)} sentences (from {len(sample)} total in 2 reports)")
    print(f"Report 1: {selected_titles[0]!r}")
    print(f"Report 2: {selected_titles[1]!r}")

    # 2. Load embeddings if RAG experiments are included
    nomic_query_embs = None
    nomic_tech_embs = None
    attack_bert_query_embs = None
    attack_bert_tech_embs = None

    nomic_q_path = EMB_DIR / "sample_191_nomic_queries.npy"
    nomic_t_path = EMB_DIR / "nomic_tech_full_with_prefixes.npy"
    ab_q_path = EMB_DIR / "sample_191_attack_bert_queries.npy"
    ab_t_path = EMB_DIR / "attack_bert_tech_full.npy"

    if 22 in args.experiments:
        nomic_query_embs = np.load(str(nomic_q_path))
        nomic_tech_embs = np.load(str(nomic_t_path))

    if any(e in args.experiments for e in [25, 27]):
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
            "bootstrap_n": BOOTSTRAP_N,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "disclaimer": (
                f"~{len(eval_sentences)}-sentence sample from 2 reports; "
                "NOT the paper's 19,178-sentence evaluation. "
                "Compare direction only."
            ),
        },
        "experiments": {},
    }

    for exp_num in args.experiments:
        if exp_num not in exp_id_map:
            continue

        exp_id = exp_id_map[exp_num]

        if exp_num == 22:
            q_embs = nomic_query_embs
            t_embs = nomic_tech_embs
        elif exp_num in (25, 27):
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
                context_windows=context_windows,
            )
        except AssertionError as e:
            print(f"\nFAILED on #{exp_num} ({exp_id}): {e}")
            raise

        paper_t = PAPER_TARGETS[exp_num]
        name = paper_t["name"]

        print_experiment_block(exp_num, name, metrics, paper_t)

        report_results["experiments"][f"exp_{exp_num}"] = {
            "name": name,
            "exp_id": exp_id,
            "metrics": metrics,
            "paper_target": paper_t,
        }

    # ── Comparison Table ──────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("SUMMARY COMPARISON TABLE (sample vs paper)")
    print("=" * 80)
    hdr = f"{'Experiment':<25} | {'P':>7} | {'R':>7} | {'F1':>7} | {'Paper F1':>9} | {'Unlisted':>8}"
    print(hdr)
    print("-" * 80)
    for exp_num in args.experiments:
        key = f"exp_{exp_num}"
        if key not in report_results["experiments"]:
            continue
        exp_data = report_results["experiments"][key]
        m = exp_data["metrics"]
        pt = exp_data["paper_target"]
        name = pt["name"]
        print(
            f"{name:<25} | {m['micro_p']:>6.2f}% | {m['micro_r']:>6.2f}% | "
            f"{m['micro_f1']:>6.2f}% | {pt['micro_f1']:>8.2f}% | {m['total_unlisted']:>8d}"
        )
    print("-" * 80)
    print("Directions to verify (sample too small for exact matching):")
    print("  - #19 BaseLLM should show lowest recall and F1.")
    print("  - #22/25/27 RAG runs should show substantially higher recall than #19.")

    # Save scored report to JSON
    with open(REPORT_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report_results, f, indent=2, ensure_ascii=False)
    print(f"\nScored metrics saved to {REPORT_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
