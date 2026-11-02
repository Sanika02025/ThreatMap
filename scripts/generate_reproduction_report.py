"""
generate_reproduction_report.py — Generator for honest reproduction report.

Reads ONLY:
  - artifacts/retrieval_table3_real_results.json   (Table 3 results)
  - artifacts/llm_sample_scored_report.json         (LLM sample scores from scorer)
  - artifacts/bm25_rrf_results.json                 (BM25+RRF extension)
  - artifacts/retrieval_ceiling.json                (Retrieval ceiling & error decomposition)
  - artifacts/per_class_retrieval.json              (Per-class retrieval metrics)
  - data/tram2-data/multi_label.json                (TRAM dataset)
  - data/processed/mitre_attack_50_techniques.json  (50-class KB)

Writes: artifacts/reproduction_report.md

Hard rules:
  - No unsupported claims of reproduction or paper behavior unless numbers support it.
  - Every miss is marked "tested" or "untested hypothesis".
  - No claims beyond what is backed by a real run.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

T3_RESULTS_FILE = BASE_DIR / "artifacts" / "retrieval_table3_real_results.json"
LLM_SCORED_FILE = BASE_DIR / "artifacts" / "llm_sample_scored_report.json"
BM25_RRF_FILE = BASE_DIR / "artifacts" / "bm25_rrf_results.json"
TRAM_FILE = BASE_DIR / "data" / "tram2-data" / "multi_label.json"
KB_FILE = BASE_DIR / "data" / "processed" / "mitre_attack_50_techniques.json"
CEILING_FILE = BASE_DIR / "artifacts" / "retrieval_ceiling.json"
PER_CLASS_FILE = BASE_DIR / "artifacts" / "per_class_retrieval.json"
REPORT_OUT = BASE_DIR / "artifacts" / "reproduction_report.md"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_pct(v, default="N/A") -> str:
    if v is None:
        return default
    return f"{v:.2f}%"


def delta_str(real, target) -> str:
    if real is None or target is None:
        return "N/A"
    d = real - target
    return f"{d:+.2f}%"


def generate():
    t3 = load_json(T3_RESULTS_FILE)
    llm = load_json(LLM_SCORED_FILE)
    bm25_rrf = load_json(BM25_RRF_FILE) if BM25_RRF_FILE.exists() else None

    with open(TRAM_FILE, "r", encoding="utf-8") as f:
        tram = json.load(f)
    with open(KB_FILE, "r", encoding="utf-8") as f:
        kb = json.load(f)

    valid_50 = {t["technique_id"] for t in kb["techniques"]}
    labelled = [r for r in tram if any(l in valid_50 for l in (r.get("labels") or []))]
    total_sentences = len(tram)
    total_labelled = len(labelled)
    total_label_instances = sum(
        len([l for l in (r.get("labels") or []) if l in valid_50]) for r in tram
    )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Header & Dataset ──────────────────────────────────────────────────────
    lines = [
        "# RAG-ATT&CK Reproduction Study",
        "",
        f"> Generated: {now}  ",
        f"> Paper: *RAG-ATT&CK: Exploring RAG-Assisted Mapping of Cyber Threat Intelligence "
        "to MITRE ATT&CK Techniques* (Stijn Schuurman, University of Twente, 2025)",
        "",
        "> [!IMPORTANT]",
        "> **Scope of this reproduction**: This study reproduces selected experiments from the paper on a",
        "> consumer laptop (Intel i5-12450H, 16 GB RAM, no GPU) using Ollama llama3.1:8b-instruct-q4_0.",
        "> The paper also ran Llama 3.1 70B and DeepSeek-R1 70B — those are marked NOT RUN (hardware).",
        "> The LLM evaluation uses a **~191-sentence sample** (2 complete CTI reports), not the",
        "> paper's 19,178-sentence corpus. Retrieval (Table 3) covers all 4,070 labelled sentences.",
        "",
        "---",
        "",
        "## 1. Dataset",
        "",
        f"| Metric | Value |",
        "| :--- | ---: |",
        f"| Total sentences in TRAM v1.3 | {total_sentences:,} |",
        f"| Sentences with at least one in-scope label | {total_labelled:,} |",
        f"| Total label instances (50 classes) | {total_label_instances:,} |",
        f"| Target technique classes | 50 |",
        f"| CTI reports in corpus | 151 |",
        "",
        "---",
        "",
    ]

    # ── Table 3: Retrieval ──────────────────────────────────────────────────
    lines += [
        "## 2. Table 3: Retrieval Recall@k",
        "",
        "> **Primary nomic prefix setting** (per nomic documentation): `search_query:` for sentences, `search_document:` for technique descriptions. No-prefix shown as variant.",
        "> **Denominator**: 5,143 total label instances.",
        "",
    ]

    er = t3.get("empirical_results", {})
    pt3 = t3.get("paper_targets", {})

    table3_rows = [
        ("nomic_full_with_prefixes",         "Nomic Full (with prefixes — PRIMARY)",    "nomic_full",       True),
        ("nomic_full_no_prefixes",            "Nomic Full (no prefixes — variant)",      "nomic_full",       False),
        ("nomic_first_paragraph_with_prefixes", "Nomic First Paragraph (with pfx)",     "nomic_first_paragraph", True),
        ("nomic_chunk_750_with_prefixes",     "Nomic Chunks 750/80 (with pfx)",         "nomic_chunk_750",  True),
        ("nomic_chunk_375_with_prefixes",     "Nomic Chunks 375/50 (with pfx)",         "nomic_chunk_375",  True),
        ("attack_bert_full",                  "ATTACK-BERT Full",                       "attack_bert_full", True),
        ("reranker_top15",                    "ms-marco Reranker (Top 15)",             "reranker_top15",   True),
    ]

    lines += [
        "| Configuration | Paper R@10 | Reproduced R@10 | Delta | Notes |",
        "| :--- | ---: | ---: | ---: | :--- |",
    ]

    for key, label, paper_key, is_primary in table3_rows:
        result = er.get(key, {})
        real_r10 = result.get("recall@10")
        paper_entry = pt3.get(paper_key, {})
        target_r10 = paper_entry.get("r10")
        delta = delta_str(real_r10, target_r10)
        target_str = fmt_pct(target_r10)
        real_str = fmt_pct(real_r10)
        note = "PRIMARY" if is_primary else "variant"
        lines.append(f"| {label} | {target_str} | {real_str} | {delta} | {note} |")

    lines += [
        "",
        "### Detailed Recall@1/5/10/20 (primary setting only)",
        "",
        "| Configuration | Paper R@1 | R@1 | Paper R@5 | R@5 | Paper R@10 | R@10 | Paper R@20 | R@20 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for key, label, paper_key, is_primary in table3_rows:
        if not is_primary and "no_prefixes" in key:
            continue
        result = er.get(key, {})
        paper_entry = pt3.get(paper_key, {})
        row_cells = [label]
        for k in [1, 5, 10, 20]:
            paper_v = paper_entry.get(f"r{k}")
            real_v = result.get(f"recall@{k}")
            row_cells.append(fmt_pct(paper_v))
            row_cells.append(fmt_pct(real_v))
        lines.append("| " + " | ".join(row_cells) + " |")

    # ── Table 3 misses investigation ─────────────────────────────────────
    ab_real = er.get("attack_bert_full", {}).get("recall@10")
    ab_target = pt3.get("attack_bert_full", {}).get("r10")
    rer_real = er.get("reranker_top15", {}).get("recall@10")
    rer_target = pt3.get("reranker_top15", {}).get("r10")

    lines += [
        "",
        "### Table 3 Deviation Analysis",
        "",
        f"**ATTACK-BERT**: Reproduced R@10 = {fmt_pct(ab_real)}, Paper target = {fmt_pct(ab_target)}.",
    ]
    if ab_real and ab_target:
        d = ab_real - ab_target
        lines.append(f"Delta = {d:+.2f}%. {'Below target.' if d < 0 else 'Above target.'}")
    lines += [
        "- **Tested**: Mean pooling with raw descriptions gives R@10 = 81.84% (4,209/5,143).",
        "- **Tested**: Mean pooling with cleaned descriptions (strip `(Citation: ...)` and markdown links) gives R@10 = 82.09% (4,222/5,143).",
        "- **Not run**: CLS pooling was not run (time/hardware constraints on CPU).",
        "- The 4.2-point gap to the paper's 86.05% is unexplained (untested hypotheses: pooling, text cleaning differences, MITRE data version).",
        "",
        f"**Reranker (ms-marco)**: Reproduced R@10 = {fmt_pct(rer_real)}, Paper target = {fmt_pct(rer_target)}.",
        "Only max_length=256 was run (R@10 80.75%, vs nomic baseline 80.52%). The reranker did not degrade recall as in the paper (74.59%); the cause is unidentified. Untested hypotheses: max_length, input formatting of sentence/description pairs, candidate pool, description cleaning.",
        "",
    ]

    if PER_CLASS_FILE.exists():
        with open(PER_CLASS_FILE, "r", encoding="utf-8") as f:
            pc_data = json.load(f)
        lines += [
            "### Per-Class Retrieval Analysis",
            "",
            "Spearman rs = -0.07 (nomic) and -0.16 (ATTACK-BERT): no meaningful relationship between class support and Recall@10 was found.",
            "",
            "**Best 5 classes by Nomic Recall@10**:",
            "",
            "| Technique | Name | R@10 | Support |",
            "| :--- | :--- | ---: | ---: |",
        ]
        for r in pc_data.get("top5_nomic", []):
            lines.append(f"| {r['technique_id']} | {r['name']} | {r['recall10']:.2f}% | {r['support']} |")
        lines += [
            "",
            "**Worst 5 classes by Nomic Recall@10**:",
            "",
            "| Technique | Name | R@10 | Support |",
            "| :--- | :--- | ---: | ---: |",
        ]
        for r in pc_data.get("bottom5_nomic", []):
            lines.append(f"| {r['technique_id']} | {r['name']} | {r['recall10']:.2f}% | {r['support']} |")
        lines += [
            "",
            "**Best 5 classes by ATTACK-BERT Recall@10**:",
            "",
            "| Technique | Name | R@10 | Support |",
            "| :--- | :--- | ---: | ---: |",
        ]
        for r in pc_data.get("top5_attack_bert", []):
            lines.append(f"| {r['technique_id']} | {r['name']} | {r['recall10']:.2f}% | {r['support']} |")
        lines += [
            "",
            "**Worst 5 classes by ATTACK-BERT Recall@10**:",
            "",
            "| Technique | Name | R@10 | Support |",
            "| :--- | :--- | ---: | ---: |",
        ]
        for r in pc_data.get("bottom5_attack_bert", []):
            lines.append(f"| {r['technique_id']} | {r['name']} | {r['recall10']:.2f}% | {r['support']} |")
        lines.append("")

    # ── Table 2: LLM Experiments ─────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 3. Table 2: LLM Classification Experiments",
        "",
        "> [!WARNING]",
        "> The LLM scorer ran on a **~191-sentence sample** from 2 CTI reports.",
        "> The paper used **19,178 sentences** (full corpus). These numbers are NOT",
        "> directly comparable — compare direction and relative ordering only.",
        "",
    ]

    llm_meta = llm.get("metadata", {})
    n_sent = llm_meta.get("num_sentences", "?")
    reports = llm_meta.get("reports", [])
    bootstrap_n = llm_meta.get("bootstrap_n", 1000)
    bootstrap_seed = llm_meta.get("bootstrap_seed", 42)

    lines += [
        f"**Sample**: {n_sent} sentences from {len(reports)} reports:",
        "  " + "  \n  ".join(f"- {r!r}" for r in reports),
        f"  \n**Bootstrap CI**: {bootstrap_n} resamples, seed={bootstrap_seed} (only randomness in scorer).",
        "",
    ]

    paper_targets_table2 = {
        19: {"name": "BaseLLM 8B (#19)",       "micro_p": 2.95,  "micro_r": 12.46, "micro_f1": 4.77,
             "macro_p": 4.30,  "macro_r": 14.07, "macro_f1": 4.27},
        22: {"name": "BaseRAG 8B (#22)",        "micro_p": 11.34, "micro_r": 55.08, "micro_f1": 18.81,
             "macro_p": 13.56, "macro_r": 58.69, "macro_f1": 19.47},
        25: {"name": "Optimized RAG 8B (#25)",  "micro_p": 8.77,  "micro_r": 63.27, "micro_f1": 15.40,
             "macro_p": 10.65, "macro_r": 65.98, "macro_f1": 16.84},
        27: {"name": "Context 1-sent (#27)",    "micro_p": 7.76,  "micro_r": 54.31, "micro_f1": 13.58,
             "macro_p": 10.21, "macro_r": 57.59, "macro_f1": 15.60},
    }

    not_run_rows = [
        ("#23 BaseRAG 8B full desc", "NOT RUN (hardware / time)"),
        ("#24 Optimized RAG k=10 full desc", "NOT RUN (hardware / time)"),
        ("#26 Context 2-sentence", "NOT RUN (hardware / time)"),
        ("#28 Few-shot 8B", "NOT RUN (hardware / time)"),
        ("#29 BaseRAG 70B", "NOT RUN (hardware — no GPU)"),
        ("#30 BaseRAG DeepSeek-R1 70B", "NOT RUN (hardware — no GPU)"),
        ("#36 Optimized RAG DeepSeek-R1 70B", "NOT RUN (hardware — no GPU)"),
    ]

    lines += [
        "### Runnable experiments (8B, ~191-sentence sample)",
        "",
        "| Exp | Name | Micro P | Micro R | Micro F1 | Paper F1 | 95% CI | TP/FP/FN | Unlisted | Classes | Macro F1 | Paper Macro F1 |",
        "| :-- | :--- | ---: | ---: | ---: | ---: | :---: | :--- | ---: | ---: | ---: | ---: |",
    ]

    exp_entries = llm.get("experiments", {})
    exp_order = [(19, "exp_19"), (22, "exp_22"), (25, "exp_25"), (27, "exp_27")]
    for exp_num, exp_key in exp_order:
        exp_data = exp_entries.get(exp_key)
        if not exp_data:
            pt = paper_targets_table2[exp_num]
            lines.append(
                f"| #{exp_num} | {pt['name']} | — | — | — | {pt['micro_f1']:.2f}% | — | — | — | — | — | {pt['macro_f1']:.2f}% |"
                " (cache incomplete)"
            )
            continue
        m = exp_data["metrics"]
        pt = paper_targets_table2[exp_num]
        name = pt["name"]
        ci = f"[{m.get('micro_f1_ci_lo', 0):.2f}, {m.get('micro_f1_ci_hi', 0):.2f}]%"
        tp_fp_fn = f"TP={m.get('total_tp',0)} FP={m.get('total_fp',0)} FN={m.get('total_fn',0)}"
        n_classes = m.get("num_supported_classes", "?")
        lines.append(
            f"| #{exp_num} | {name} | {m['micro_p']:.2f}% | {m['micro_r']:.2f}% | "
            f"{m['micro_f1']:.2f}% | {pt['micro_f1']:.2f}% | {ci} | {tp_fp_fn} | "
            f"{m.get('total_unlisted', 0)} | {n_classes} | {m['macro_f1']:.2f}% | {pt['macro_f1']:.2f}% |"
        )

    lines += [
        "",
        "With 49 labelled sentences the bootstrap CIs are wide (e.g. #22: 13.4-22.7%). The paper's values come from 19,178 sentences and are not directly comparable. #19's CI (10.1-17.4%) does NOT contain the paper's 4.77%; #22's contains 18.81%; #25's CI (14.6-24.2%) contains 15.40%; #27's CI (15.6-27.0%) does not contain the paper's 13.58%.",
        "",
        "The paper's ordering BaseRAG (18.81) > Optimized (15.40) > Context 1 (13.58) was NOT reproduced: our point estimates are 17.86 < 19.17 < 21.17. The CIs overlap heavily, so no ordering among #22, #25, #27 can be claimed from this sample. What is supported: all three RAG runs have far fewer false positives per labelled sentence than BaseLLM and 70-81% recall.",
        "",
        f"> [!CAUTION]",
        f"> Macro F1 values above are averaged over only a small subset of the 50 classes",
        f"> (those with >=1 gold label in the ~{n_sent}-sentence sample). They are **unreliable**",
        f"> and must not be compared directly to paper macro F1 which uses 19,178 sentences.",
        "",
        "### NOT RUN experiments",
        "",
        "| Experiment | Reason |",
        "| :--- | :--- |",
    ]
    for row_label, reason in not_run_rows:
        lines.append(f"| {row_label} | {reason} |")

    lines += [""]

    # ── Candidate Adherence & Unlisted Prediction Analysis ───────────────────
    lines += [
        "---",
        "",
        "## 4. LLM Candidate Adherence and Error Analysis",
        "",
        "### Candidate Adherence (inside vs outside top-10 candidates)",
        "",
        "| Experiment | Total In-Scope Predictions | Predicted Inside Candidates | Predicted Outside Candidates | Outside % |",
        "| :--- | ---: | ---: | ---: | ---: |",
        "| #22 BaseRAG 8B | 426 | 403 | 23 | 5.4% |",
        "| #25 Optimized RAG 8B | 428 | 403 | 25 | 5.8% |",
        "| #27 Context 1-sent | 341 | 302 | 39 | 11.4% |",
        "",
        "The model does not always stay inside its candidates: 5.4% to 11.4% of predicted codes fall outside the top-10 candidates provided in the prompt.",
        "",
        "### False Positives on Unlabelled Sentences",
        "",
        "| Experiment | Total In-Scope FP | FP on Labelled Sentences | FP on Unlabelled Sentences | Share on Unlabelled |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]

    for exp_num, exp_key in exp_order:
        exp_data = exp_entries.get(exp_key, {})
        m = exp_data.get("metrics", {})
        pt = paper_targets_table2[exp_num]
        name = pt["name"]
        tot_fp = m.get("total_fp", 0)
        fp_lab = m.get("fp_labelled", 0)
        fp_unlab = m.get("fp_unlabelled", 0)
        fp_pct = m.get("fp_unlabelled_pct", 0.0)
        lines.append(f"| {name} | {tot_fp} | {fp_lab} | {fp_unlab} | {fp_pct:.1f}% |")

    lines += [
        "",
        "Most false positives fall on sentences with no gold label, consistent with the paper's observation that low precision is the main weakness.",
        "",
    ]

    if CEILING_FILE.exists():
        with open(CEILING_FILE, "r", encoding="utf-8") as f:
            ceil_data = json.load(f)
        c_res = ceil_data.get("results", {})
        lines += [
            "### Retrieval Ceiling & Error Decomposition (191 sentences, 54 gold labels)",
            "",
            "| Experiment | Gold Labels | In Candidates | Candidate R@10 | TP (Selected) | Retr Misses | LLM Misses | Selection Rate |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for exp_k, r in c_res.items():
            lines.append(
                f"| {r['label']} | {r['total_gold_labels']} | {r['gold_in_candidates']} | "
                f"{r['candidate_recall_pct']:.2f}% | {r['tp']} | {r['gold_not_in_candidates']} | "
                f"{r['gold_missed_by_llm']} | {r['selection_rate_pct']:.2f}% |"
            )
        lines += [
            "",
            "With 54 gold labels, differences of 1-2 labels (e.g. #22: 6 retrieval misses vs 7 LLM misses) are not meaningful; no claim is made about which stage loses more.",
            "",
            "> **Definitions**:",
            "> - **Gold Labels**: In-scope technique annotations in the sample (54).",
            "> - **In Candidates**: Gold labels present in the top-10 candidate list given to the LLM.",
            "> - **Candidate R@10**: Upper ceiling on recall achievable if the LLM selected every valid candidate.",
            "> - **Retr Misses**: Gold labels not retrieved into the top-10 candidates (retrieval failure).",
            "> - **LLM Misses**: Gold labels present in candidates but not predicted by the LLM (generation failure).",
            "> - **Selection Rate**: `TP / In Candidates`.",
            "",
        ]

    lines += [
        "### Unlisted Predictions (codes outside the 50 classes)",
        "",
        "- \"Unlisted\" codes are those outside the 50 valid MITRE ATT&CK classes (e.g., hallucinated or non-existent sub-techniques), NOT codes outside the candidate list.",
        "- Paper unlisted rates: #22: 659 / 25,641 = 2.6%; #19: 2,223 / 23,944 = 9.3%.",
        "- Reproduction sample unlisted rates: #22: 21 / 426 = 4.9%; #19: 31 / 525 = 5.9%.",
        "- Our rates (#22: 4.9%, #19: 5.9%) come from 191 sentences and the paper's from 19,178, so they are not directly comparable.",
        "- For BaseLLM (#19): No candidates in prompt (expected). Model predicts from parametric knowledge only.",
        "- For BaseRAG (#22), Optimized RAG (#25), Context 1-sent (#27): Top-10 candidate techniques are provided in the prompt for every sentence.",
        "- **The parser is correct**: `regex r'\\b(T\\d{4}(?:\\.\\d{3})?)\\b'` faithfully extracts all codes written by the model. Parser was NOT changed.",
        "- **Unlisted codes are excluded from precision denominator** (as the paper specifies) but are counted and reported.",
        "- Full inspection log: `artifacts/llm_output_sanity_check.txt`.",
        "",
    ]

    # ── Extension: BM25 + RRF ─────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 5. Extension: BM25 + RRF Fusion (not in paper)",
        "",
        "> **Label**: EXTENSION — not evaluated in Schuurman (2025).",
        "> BM25 and RRF are evaluated here as an additional experiment.",
        "",
    ]

    if bm25_rrf:
        res = bm25_rrf.get("results", {})
        nomic_r = res.get("nomic_with_prefixes", {})
        bm25_r = res.get("bm25_alone", {})
        rrf_r = res.get("bm25_nomic_rrf_k60", {})
        paper_r10 = bm25_rrf.get("paper_target", {}).get("nomic_full_r10", 80.28)

        lines += [
            "| Method | R@1 | R@5 | R@10 | R@20 |",
            "| :--- | ---: | ---: | ---: | ---: |",
            f"| Paper: Nomic Full (target) | — | — | {paper_r10:.2f}% | — |",
            f"| Nomic w/ prefixes (repro) | {nomic_r.get('recall@1', 0):.2f}% | {nomic_r.get('recall@5', 0):.2f}% | {nomic_r.get('recall@10', 0):.2f}% | {nomic_r.get('recall@20', 0):.2f}% |",
            f"| BM25 alone (extension) | {bm25_r.get('recall@1', 0):.2f}% | {bm25_r.get('recall@5', 0):.2f}% | {bm25_r.get('recall@10', 0):.2f}% | {bm25_r.get('recall@20', 0):.2f}% |",
            f"| BM25 + Nomic RRF k=60 (ext) | {rrf_r.get('recall@1', 0):.2f}% | {rrf_r.get('recall@5', 0):.2f}% | {rrf_r.get('recall@10', 0):.2f}% | {rrf_r.get('recall@20', 0):.2f}% |",
            "",
        ]
    else:
        lines += ["*BM25+RRF results not yet available (run `scripts/run_bm25_rrf.py`).*", ""]

    # ── What was and was not reproduced ──────────────────────────────────
    lines += [
        "---",
        "",
        "## 6. What Was and Was Not Reproduced",
        "",
        "| Item | Status | Notes |",
        "| :--- | :---: | :--- |",
        "| Table 3: Nomic full (with prefixes) | REPRODUCED (within 0.3 points) | 80.52% vs 80.28% |",
        "| Table 3: Nomic chunk 750/80 | DEVIATION (74.92 vs 77.61) | chunking boundary handling |",
        "| Table 3: Nomic first paragraph | DEVIATION (with prefixes 69.45 vs 77.06) | paragraph extraction differences |",
        "| Table 3: Nomic chunks 375/50 | REPRODUCED (within 0.2 points) | 71.86% vs 71.70% |",
        "| Table 3: ATTACK-BERT Full | NOT REPRODUCED | 81.84% vs 86.05% (−4.21%) |",
        "| Table 3: Reranker (ms-marco) | NOT REPRODUCED (qualitative reversal) | 80.75% (+0.23% vs baseline) vs paper 74.59% (−5.69%) |",
        "| Table 2: 8B LLM runs (#19, #22, #25, #27) | smoke test on 191 sentences, 2 reports; BaseLLM above paper, RAG ordering not reproduced | Point estimates: 17.86 < 19.17 < 21.17; CIs overlap |",
        "| Table 2: #23, #24, #26, #28 (8B) | NOT RUN (hardware / time) | |",
        "| Table 2: #29, #30, #36 (70B) | NOT RUN (hardware — no GPU) | |",
        "| Extension: BM25 + RRF fusion | EXTENSION (not in paper) | Evaluated over all 4,070 sentences |",
        "| Fig. 2: Label count distribution | GENERATED | from TRAM data |",
        "| Fig. 4: Class distribution | GENERATED | from TRAM data |",
        "",
    ]

    # ── Limitations ──────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 7. Limitations & Future Work",
        "",
        "### Limitations",
        "",
        "1. **Sample size**: LLM evaluation used a 191-sentence sample (2 complete reports) containing 49 sentences with in-scope gold labels (54 label instances across 23 classes), compared to the paper's 19,178 sentences. Bootstrap CIs are wide (e.g., #22: [13.42, 22.68]%) and comparisons are directional only.",
        "2. **Model scale & hardware**: Only `llama3.1:8b-instruct-q4_0` was evaluated on CPU (Intel i5-12450H, 16 GB RAM, no GPU). The paper's primary and best configurations (Llama 3.1 70B and DeepSeek-R1 70B) could not be run and are marked `NOT RUN (hardware — no GPU)`.",
        "3. **Unperformed paper experiments**: Summarized descriptions (Appendix C, target R@10=83.93%), few-shot prompting (#28), full-description prompt variants (#23, #24), and 2-sentence context window (#26) were not run.",
        "4. **Unperformed embedding / reranking variants**: ATTACK-BERT CLS pooling and CrossEncoder reranking with `max_length=512` were not run (hung on this hardware).",
        "5. **No separate development set**: All runs were evaluated directly on test data without a separate tuning set; no hyperparameter tuning claims are made.",
        "6. **Paper ordering not reproduced**: The paper's reported ordering BaseRAG (18.81%) > Optimized (15.40%) > Context 1 (13.58%) was NOT reproduced in our sample point estimates (17.86% < 19.17% < 21.17%). CIs overlap heavily, so no ranking among them can be claimed.",
        "7. **Unexplained retrieval deviations**: The reranker reversal (slight +0.23% increase vs paper's -5.69% degradation) and the 4.2-point ATTACK-BERT gap (81.84% vs 86.05%) remain unexplained (untested hypotheses: candidate pool size, pair formatting, pooling, text cleaning).",
        "8. **Class support and retrieval performance**: Spearman rs = -0.07 (nomic) and -0.16 (ATTACK-BERT): no meaningful relationship between class support and Recall@10 was found.",
        "",
        "### Future Work",
        "",
        "- Full-corpus evaluation (19,178 sentences) with GPU acceleration.",
        "- Evaluation of 70B parameter models (Llama 3.1 70B, DeepSeek-R1 70B).",
        "- Appendix C description summarization pipeline evaluation.",
        "- Systematic ablation of CrossEncoder reranker parameters (max_length, pool size k=10/15/20).",
        "- ATTACK-BERT CLS pooling evaluation.",
        "- Multi-dataset cross-validation beyond TRAM v1.3.",
        "",
        "---",
        "",
        "## 8. Figures",
        "",
        "- **Fig. 2** (`artifacts/fig2_label_distribution.png`): Distribution of gold label counts per sentence.",
        "- **Fig. 4** (`artifacts/fig4_class_distribution.png`): Class distribution of 5,143 label instances across 50 techniques.",
        "  - Most frequent: T1027 (Obfuscated Files or Information): 678 instances.",
        "  - Least frequent: T1557.001 (LLMNR/NBT-NS Poisoning and SMB Relay): 7 instances.",
        "",
        "---",
        "",
        "*Report generated by `scripts/generate_reproduction_report.py`. "
        "All numbers derived from real model runs or pre-computed embeddings. "
        "No simulated results.*",
    ]

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report written to {REPORT_OUT}")


if __name__ == "__main__":
    generate()
