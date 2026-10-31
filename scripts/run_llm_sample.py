"""
run_llm_sample.py — Real LLM inference for RAG-ATT&CK Table 2 reproduction.

Runs experiments #19 (BaseLLM), #22 (BaseRAG), #25 (Optimized), #27 (Context-1)
on a fixed 2-report sample using llama3.1:8b-instruct-q4_0 via Ollama.

Hard rules:
  - No simulation. Every response comes from Ollama or raises RuntimeError.
  - Cache every raw reply to artifacts/llm_cache/<exp_id>.jsonl (resumable).
  - Scorer (score_llm_sample.py) must FAIL if any cache entry is missing.
  - Temperature=0, seed=42, num_predict=64, num_ctx=4096 for all runs.

Usage:
  python scripts/run_llm_sample.py --mode pilot        # 20 sentences, timing
  python scripts/run_llm_sample.py --mode sample       # full 2-report sample, all 4 experiments
  python scripts/run_llm_sample.py --mode sample --experiments 19 22  # specific experiments
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import requests

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "tram2-data" / "multi_label.json"
KB_FILE = ROOT / "data" / "processed" / "mitre_attack_50_techniques.json"
EMB_DIR = ROOT / "artifacts" / "embedding_cache"
CACHE_DIR = ROOT / "artifacts" / "llm_cache"
RESULTS_FILE = ROOT / "artifacts" / "llm_sample_results.json"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── model / Ollama settings ───────────────────────────────────────────────────
MODEL = "llama3.1:8b-instruct-q4_0"
OLLAMA_URL = "http://localhost:11434/api/chat"
OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 64, "num_ctx": 4096}

# ── sample selection (fixed, reproducible) ────────────────────────────────────
# Sort all doc_titles alphabetically; take doc index 0 and index 4.
# doc[0]: '3CXDesktopApp Backdoored...'         — 81 rows, 15 in-scope labelled
# doc[4]: 'AA21076A TrickBot Malware'            — 110 rows, 34 in-scope labelled
# Total: 191 sentences, 49 in-scope labelled rows
SELECTED_DOC_INDICES = [0, 4]

# Number of retrieval candidates for RAG experiments
RAG_K = 10


# ── cache helpers ─────────────────────────────────────────────────────────────

def _cache_key(model: str, messages: List[Dict], options: Dict) -> str:
    payload = {"model": model, "messages": messages, "options": options}
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_cache(exp_id: str) -> Dict[str, Dict]:
    """Load all cache entries for an experiment into {hash: entry} dict."""
    cache_file = CACHE_DIR / f"{exp_id}.jsonl"
    cache: Dict[str, Dict] = {}
    if not cache_file.exists():
        return cache
    with open(cache_file, "r", encoding="utf-8") as f:
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


def _append_cache(exp_id: str, entry: Dict) -> None:
    """Append a single cache entry as a JSONL line."""
    cache_file = CACHE_DIR / f"{exp_id}.jsonl"
    with open(cache_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Ollama call ───────────────────────────────────────────────────────────────

def call_ollama(messages: List[Dict[str, str]]) -> Tuple[str, float]:
    """
    Call Ollama /api/chat. Returns (content, duration_seconds).
    Raises RuntimeError if Ollama is unreachable or returns an error.
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": OPTIONS,
    }
    t0 = time.time()
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Ollama unreachable at {OLLAMA_URL}. "
            "Make sure `ollama serve` is running. Error: {exc}"
        ) from exc
    duration = time.time() - t0

    if resp.status_code != 200:
        raise RuntimeError(
            f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}"
        )
    data = resp.json()
    content = data.get("message", {}).get("content", "")
    return content, duration


def call_ollama_cached(
    exp_id: str,
    messages: List[Dict[str, str]],
    cache: Dict[str, Dict],
) -> Tuple[str, float, bool]:
    """
    Return (content, duration_s, was_cached).
    Writes new entries to cache dict and appends to JSONL file.
    """
    key = _cache_key(MODEL, messages, OPTIONS)
    if key in cache:
        entry = cache[key]
        return entry["response"], entry.get("duration_s", 0.0), True

    content, duration = call_ollama(messages)
    entry = {
        "hash": key,
        "model": MODEL,
        "messages": messages,
        "options": OPTIONS,
        "response": content,
        "duration_s": duration,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    cache[key] = entry
    _append_cache(exp_id, entry)
    return content, duration, False


# ── data loading ──────────────────────────────────────────────────────────────

def load_tram_data() -> List[Dict]:
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_kb() -> Tuple[List[Dict], Set[str]]:
    with open(KB_FILE, encoding="utf-8") as f:
        kb = json.load(f)
    techs = kb["techniques"]
    classes50 = {t["technique_id"] for t in techs}
    return techs, classes50


def select_sample(data: List[Dict]) -> Tuple[List[Dict], List[str]]:
    """Return (sample_rows, selected_doc_titles) using fixed doc indices."""
    docs = sorted(set(r["doc_title"] for r in data))
    selected_titles = [docs[i] for i in SELECTED_DOC_INDICES]
    sample = [r for r in data if r["doc_title"] in set(selected_titles)]
    # Preserve original ordering (file order within each doc)
    return sample, selected_titles


# ── retrieval helpers ─────────────────────────────────────────────────────────

def cosine_similarity_matrix(query_embs: np.ndarray, tech_embs: np.ndarray) -> np.ndarray:
    """Compute cosine similarities. Returns (n_queries, n_techs)."""
    qn = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-9)
    tn = tech_embs / (np.linalg.norm(tech_embs, axis=1, keepdims=True) + 1e-9)
    return qn @ tn.T


def get_top_k_techniques(
    sentence_idx_in_labelled: int,
    tech_embs: np.ndarray,
    query_embs: np.ndarray,
    techniques: List[Dict],
    k: int = 10,
) -> List[Dict]:
    """Return top-k technique dicts by cosine similarity."""
    sims = cosine_similarity_matrix(
        query_embs[sentence_idx_in_labelled : sentence_idx_in_labelled + 1],
        tech_embs,
    )[0]
    top_indices = np.argsort(sims)[::-1][:k]
    return [techniques[i] for i in top_indices]


# ── embedding index for full dataset ─────────────────────────────────────────

def build_sentence_index(data: List[Dict]) -> Dict[str, int]:
    """
    Build {sentence -> index in labelled array} from the FULL dataset order
    (same order as embedding cache was built in run_retrieval_table3.py).
    Only labelled sentences are in the embedding cache.
    """
    labelled = [r for r in data if r.get("labels")]
    return {r["sentence"]: i for i, r in enumerate(labelled)}


# ── prompt builders ───────────────────────────────────────────────────────────

def _tech_list_for_base_llm(techniques: List[Dict]) -> str:
    """All 50 technique IDs + names, one per line — Appendix B."""
    return "\n".join(f"{t['technique_id']} : {t['technique_name']}" for t in techniques)


def build_base_llm_messages(sentence: str, techniques: List[Dict]) -> List[Dict[str, str]]:
    """
    Experiment #19 — BaseLLM (Appendix B).
    Single user message containing all 50 technique IDs so the model
    knows what the valid outputs are.
    """
    tech_list = _tech_list_for_base_llm(techniques)
    content = (
        "You are a cybersecurity expert specializing in MITRE ATT&CK framework mapping.\n\n"
        "Given a sentence from a Cyber Threat Intelligence (CTI) report, "
        "identify the relevant MITRE ATT&CK techniques from the list below.\n\n"
        "Available techniques:\n"
        f"{tech_list}\n\n"
        "Provide your answer as a list of technique identifiers and names in the format:\n"
        "Txxxx : Name\n"
        "If no techniques apply, output 'No Mapping'.\n\n"
        f"Sentence: {sentence}"
    )
    return [{"role": "user", "content": content}]


def _format_rag_candidates(candidates: List[Dict]) -> str:
    """Format candidates as '[technique: Txxxx - Name]' — Appendix D / paper Section 5."""
    return "\n".join(
        f"[technique: {t['technique_id']} - {t['technique_name']}]"
        for t in candidates
    )


def build_base_rag_messages(
    sentence: str, candidates: List[Dict]
) -> List[Dict[str, str]]:
    """
    Experiment #22 — BaseRAG (Appendix D, nomic k=10).
    """
    cands_str = _format_rag_candidates(candidates)
    content = (
        "You are a cybersecurity expert specializing in MITRE ATT&CK framework mapping.\n\n"
        "Given a sentence from a Cyber Threat Intelligence (CTI) report and "
        "candidate MITRE ATT&CK techniques, identify which technique(s) apply.\n\n"
        "Candidate Techniques:\n"
        f"{cands_str}\n\n"
        "Provide your answer as a list of technique identifiers and names in the format:\n"
        "Txxxx : Name\n"
        "If no techniques apply, output 'No Mapping'.\n\n"
        f"Sentence: {sentence}"
    )
    return [{"role": "user", "content": content}]


def build_optimized_rag_messages(
    sentence: str, candidates: List[Dict]
) -> List[Dict[str, str]]:
    """
    Experiment #25 — Optimized RAG (ATTACK-BERT k=10, Appendix D variant).
    """
    cands_str = _format_rag_candidates(candidates)
    content = (
        "You are a cybersecurity expert specializing in MITRE ATT&CK framework mapping.\n\n"
        "Given a sentence from a Cyber Threat Intelligence (CTI) report and the "
        f"top-{RAG_K} candidate MITRE ATT&CK techniques, "
        "identify which technique(s) are described in the sentence.\n\n"
        "Candidate Techniques:\n"
        f"{cands_str}\n\n"
        "Rules:\n"
        "- Select only technique(s) that are directly evidenced in the sentence.\n"
        "- Provide your answer as a list of technique identifiers and names in the format:\n"
        "  Txxxx : Name\n"
        "- If no techniques apply, output 'No Mapping'.\n\n"
        f"Sentence: {sentence}"
    )
    return [{"role": "user", "content": content}]


def build_context1_messages(
    sentence: str,
    candidates: List[Dict],
    prev_sentence: Optional[str],
    next_sentence: Optional[str],
) -> List[Dict[str, str]]:
    """
    Experiment #27 — Context 1 sentence (Appendix E + Section 5-F3 user template).
    System prompt + user message with context window.
    """
    cands_str = _format_rag_candidates(candidates)

    context_parts = []
    if prev_sentence:
        context_parts.append(f"[Previous]: {prev_sentence}")
    context_parts.append(f"[Target]: {sentence}")
    if next_sentence:
        context_parts.append(f"[Next]: {next_sentence}")
    context_block = "\n".join(context_parts)

    system_content = (
        "You are a cybersecurity expert specializing in MITRE ATT&CK framework mapping. "
        "Given a target sentence and its surrounding context from a Cyber Threat Intelligence "
        "(CTI) report along with candidate MITRE ATT&CK techniques, identify which technique(s) "
        "apply to the target sentence."
    )
    user_content = (
        f"Surrounding Context:\n{context_block}\n\n"
        f"Candidate Techniques:\n{cands_str}\n\n"
        "Provide your answer as a list of technique identifiers and names in the format:\n"
        "Txxxx : Name\n"
        "If no techniques apply, output 'No Mapping'.\n\n"
        f"Target Sentence: {sentence}"
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# ── context window helper ─────────────────────────────────────────────────────

def build_context_windows(sample: List[Dict]) -> Dict[int, Tuple[Optional[str], Optional[str]]]:
    """
    Build {row_index -> (prev_sentence, next_sentence)} using doc-boundary awareness.
    Sentences from different doc_titles are not considered neighbours.
    """
    windows: Dict[int, Tuple[Optional[str], Optional[str]]] = {}
    for i, row in enumerate(sample):
        prev_sent = None
        next_sent = None
        if i > 0 and sample[i - 1]["doc_title"] == row["doc_title"]:
            prev_sent = sample[i - 1]["sentence"]
        if i < len(sample) - 1 and sample[i + 1]["doc_title"] == row["doc_title"]:
            next_sent = sample[i + 1]["sentence"]
        windows[i] = (prev_sent, next_sent)
    return windows


# ── main runner ───────────────────────────────────────────────────────────────

def run_experiment(
    exp_id: str,
    exp_num: int,
    sentences: List[Dict],          # sample rows (all sentences incl. unlabelled)
    techniques: List[Dict],
    tech_embs: Optional[np.ndarray],   # for RAG experiments
    query_embs: Optional[np.ndarray],  # pre-computed per labelled sentence
    sent_to_labelled_idx: Dict[str, int],
    context_windows: Optional[Dict[int, Tuple[Optional[str], Optional[str]]]],
    pilot_only: bool = False,
    verbose: bool = True,
) -> List[Dict]:
    """
    Run one experiment. Returns list of result records.
    Caches every call and resumes from existing cache.
    """
    cache = _load_cache(exp_id)
    n_cached = len(cache)
    print(f"\n{'='*60}")
    print(f"Experiment #{exp_num} ({exp_id}): {len(sentences)} sentences")
    print(f"  Cached entries found: {n_cached}")

    rows_to_run = sentences[:20] if pilot_only else sentences
    results = []

    for i, row in enumerate(rows_to_run):
        sentence = row["sentence"]
        doc_title = row["doc_title"]

        # Build messages depending on experiment
        if exp_num == 19:  # BaseLLM
            messages = build_base_llm_messages(sentence, techniques)

        elif exp_num in (22, 25, 27):
            # Need embeddings for retrieval
            lab_idx = sent_to_labelled_idx.get(sentence)
            if lab_idx is None:
                # Unlabelled sentence — no embedding. Use empty candidates.
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

        content, duration, was_cached = call_ollama_cached(exp_id, messages, cache)

        if verbose or pilot_only:
            cached_marker = "[cached]" if was_cached else f"[{duration:.1f}s]"
            print(f"  [{i+1:3d}/{len(rows_to_run)}] {cached_marker} {repr(content[:80])}", flush=True)

        results.append({
            "row_index": i,
            "doc_title": doc_title,
            "sentence": sentence,
            "gold_labels": row.get("labels") or [],
            "raw_response": content,
            "duration_s": duration,
            "was_cached": was_cached,
            "exp_id": exp_id,
            "exp_num": exp_num,
        })

    new_calls = sum(1 for r in results if not r["was_cached"])
    if new_calls > 0:
        durations = [r["duration_s"] for r in results if not r["was_cached"]]
        avg_dur = sum(durations) / len(durations)
        print(f"  New calls: {new_calls}, avg {avg_dur:.2f}s/call, total {sum(durations):.1f}s", flush=True)
    else:
        print("  All calls served from cache.", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run real LLM inference for RAG-ATT&CK Table 2")
    parser.add_argument("--mode", choices=["pilot", "sample"], default="sample",
                        help="pilot=first 20 sentences only; sample=full 2-report sample")
    parser.add_argument("--experiments", nargs="+", type=int,
                        default=[19, 22, 25, 27],
                        help="Which experiments to run (19 22 25 27)")
    args = parser.parse_args()

    print("Loading TRAM data...")
    data = load_tram_data()
    sample, selected_titles = select_sample(data)
    print(f"Sample: {len(sample)} sentences from {len(selected_titles)} reports")
    print(f"  Doc 0: {selected_titles[0]!r}")
    print(f"  Doc 1: {selected_titles[1]!r}")

    print("\nLoading knowledge base (50 techniques)...")
    techniques, classes50 = load_kb()
    print(f"  {len(techniques)} techniques, {len(classes50)} class IDs")

    print("\nBuilding sentence->labelled-index map (for retrieval)...")
    sent_to_labelled_idx = build_sentence_index(data)

    print("Building context windows for sample...")
    context_windows = build_context_windows(sample)

    # Load embeddings
    print("\nLoading embedding caches...")
    nomic_query_embs = None
    nomic_tech_embs = None
    attack_bert_query_embs = None
    attack_bert_tech_embs = None

    nomic_q_path = EMB_DIR / "nomic_queries_with_prefixes.npy"
    nomic_t_path = EMB_DIR / "nomic_tech_full_with_prefixes.npy"
    ab_q_path = EMB_DIR / "attack_bert_queries.npy"
    ab_t_path = EMB_DIR / "attack_bert_tech_full.npy"

    needs_nomic = any(e in args.experiments for e in [22, 27])
    needs_ab = 25 in args.experiments

    if needs_nomic:
        if nomic_q_path.exists() and nomic_t_path.exists():
            nomic_query_embs = np.load(str(nomic_q_path))
            nomic_tech_embs = np.load(str(nomic_t_path))
            print(f"  Nomic: query={nomic_query_embs.shape}, tech={nomic_tech_embs.shape}")
        else:
            raise FileNotFoundError(
                f"Nomic embeddings missing. Run scripts/run_retrieval_table3.py first.\n"
                f"Expected: {nomic_q_path}"
            )

    if needs_ab:
        if ab_q_path.exists() and ab_t_path.exists():
            attack_bert_query_embs = np.load(str(ab_q_path))
            attack_bert_tech_embs = np.load(str(ab_t_path))
            print(f"  ATTACK-BERT: query={attack_bert_query_embs.shape}, tech={attack_bert_tech_embs.shape}")
        else:
            raise FileNotFoundError(
                f"ATTACK-BERT embeddings missing. Run scripts/run_retrieval_table3.py first.\n"
                f"Expected: {ab_q_path}"
            )

    # Verify Ollama is reachable (fail fast before spending time on prompts)
    print("\nChecking Ollama availability...")
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        if MODEL not in models:
            raise RuntimeError(
                f"Model '{MODEL}' not found in Ollama. Available: {models}"
            )
        print(f"  Ollama OK — model '{MODEL}' available")
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError("Ollama is not running. Start with `ollama serve`.") from exc

    # Map experiment number → experiment ID string
    exp_id_map = {
        19: "exp19_base_llm",
        22: "exp22_base_rag",
        25: "exp25_optimized_rag",
        27: "exp27_context1",
    }

    pilot_only = (args.mode == "pilot")
    all_results: Dict[int, List[Dict]] = {}

    for exp_num in args.experiments:
        if exp_num not in exp_id_map:
            print(f"WARNING: Unknown experiment number {exp_num}, skipping")
            continue

        exp_id = exp_id_map[exp_num]

        # Select embeddings
        if exp_num in (22, 27):
            q_embs = nomic_query_embs
            t_embs = nomic_tech_embs
        elif exp_num == 25:
            q_embs = attack_bert_query_embs
            t_embs = attack_bert_tech_embs
        else:
            q_embs = None
            t_embs = None

        results = run_experiment(
            exp_id=exp_id,
            exp_num=exp_num,
            sentences=sample,
            techniques=techniques,
            tech_embs=t_embs,
            query_embs=q_embs,
            sent_to_labelled_idx=sent_to_labelled_idx,
            context_windows=context_windows,
            pilot_only=pilot_only,
            verbose=(args.mode == "pilot"),
        )
        all_results[exp_num] = results

    # Save metadata
    metadata = {
        "mode": args.mode,
        "model": MODEL,
        "options": OPTIONS,
        "selected_doc_indices": SELECTED_DOC_INDICES,
        "selected_doc_titles": selected_titles,
        "total_sample_sentences": len(sample),
        "sentences_run": 20 if pilot_only else len(sample),
        "experiments_run": args.experiments,
        "exp_id_map": exp_id_map,
        "rag_k": RAG_K,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nMetadata saved to {RESULTS_FILE}")
    print("\nDone. Run `python scripts/score_llm_sample.py` to compute metrics.")


if __name__ == "__main__":
    main()

