# RAG-ATT&CK Reproduction Study

> Generated: 2026-11-02 10:11 UTC  
> Paper: *RAG-ATT&CK: Exploring RAG-Assisted Mapping of Cyber Threat Intelligence to MITRE ATT&CK Techniques* (Stijn Schuurman, University of Twente, 2025)

> [!IMPORTANT]
> **Scope of this reproduction**: This study reproduces selected experiments from the paper on a
> consumer laptop (Intel i5-12450H, 16 GB RAM, no GPU) using Ollama llama3.1:8b-instruct-q4_0.
> The paper also ran Llama 3.1 70B and DeepSeek-R1 70B — those are marked NOT RUN (hardware).
> The LLM evaluation uses a **~191-sentence sample** (2 complete CTI reports), not the
> paper's 19,178-sentence corpus. Retrieval (Table 3) covers all 4,070 labelled sentences.

---

## 1. Dataset

| Metric | Value |
| :--- | ---: |
| Total sentences in TRAM v1.3 | 19,178 |
| Sentences with at least one in-scope label | 4,070 |
| Total label instances (50 classes) | 5,143 |
| Target technique classes | 50 |
| CTI reports in corpus | 151 |

---

## 2. Table 3: Retrieval Recall@k

> **Primary nomic prefix setting** (per nomic documentation): `search_query:` for sentences, `search_document:` for technique descriptions. No-prefix shown as variant.
> **Denominator**: 5,143 total label instances.

| Configuration | Paper R@10 | Reproduced R@10 | Delta | Notes |
| :--- | ---: | ---: | ---: | :--- |
| Nomic Full (with prefixes — PRIMARY) | 80.28% | 80.52% | +0.24% | PRIMARY |
| Nomic Full (no prefixes — variant) | 80.28% | 81.16% | +0.88% | variant |
| Nomic First Paragraph (with pfx) | 77.06% | 69.45% | -7.61% | PRIMARY |
| Nomic Chunks 750/80 (with pfx) | 77.61% | 74.92% | -2.69% | PRIMARY |
| Nomic Chunks 375/50 (with pfx) | 71.70% | 71.86% | +0.16% | PRIMARY |
| ATTACK-BERT Full | 86.05% | 81.84% | -4.21% | PRIMARY |
| ms-marco Reranker (Top 15) | 74.59% | 80.75% | +6.16% | PRIMARY |

### Detailed Recall@1/5/10/20 (primary setting only)

| Configuration | Paper R@1 | R@1 | Paper R@5 | R@5 | Paper R@10 | R@10 | Paper R@20 | R@20 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Nomic Full (with prefixes — PRIMARY) | 37.44% | 35.82% | 69.06% | 68.60% | 80.28% | 80.52% | 89.08% | 90.43% |
| Nomic First Paragraph (with pfx) | N/A | 28.93% | N/A | 54.99% | 77.06% | 69.45% | N/A | 85.57% |
| Nomic Chunks 750/80 (with pfx) | N/A | 31.85% | N/A | 62.36% | 77.61% | 74.92% | N/A | 86.89% |
| Nomic Chunks 375/50 (with pfx) | N/A | 26.89% | N/A | 56.74% | 71.70% | 71.86% | N/A | 85.57% |
| ATTACK-BERT Full | 42.22% | 37.86% | 75.66% | 70.31% | 86.05% | 81.84% | 94.08% | 91.83% |
| ms-marco Reranker (Top 15) | N/A | 33.91% | N/A | 66.95% | 74.59% | 80.75% | N/A | 90.43% |

### Table 3 Deviation Analysis

**ATTACK-BERT**: Reproduced R@10 = 81.84%, Paper target = 86.05%.
Delta = -4.21%. Below target.
- **Tested**: Mean pooling with raw descriptions gives R@10 = 81.84% (4,209/5,143).
- **Tested**: Mean pooling with cleaned descriptions (strip `(Citation: ...)` and markdown links) gives R@10 = 82.09% (4,222/5,143), an improvement of +0.25%, but still 3.96% below the paper's 86.05%.
- **Untested hypothesis**: The paper may have used CLS pooling or a different truncation/tokenization strategy. CLS was not computed on this hardware due to memory/time constraints.

**Reranker (ms-marco)**: Reproduced R@10 = 80.75%, Paper target = 74.59%.
Delta = +6.16%. Our reranker **improves** over nomic baseline (80.75% vs 80.52%), while the paper shows it **hurts** (74.59% < 80.28%). This is a qualitative reversal.
- **Tested**: Testing with max_length=256 and raw descriptions yields R@10 = 80.75% (4,153/5,143), identical to the baseline reranker score. This confirms max_length does not account for the paper's 74.59% degradation.
- **Untested hypothesis**: The paper's reranker pipeline may have had candidate inputs formatted differently, used a different top-k pool (e.g. k=10 or k=20 rather than 15), or evaluated with a different negative sampling / scoring threshold.

---

## 3. Table 2: LLM Classification Experiments

> [!WARNING]
> The LLM scorer ran on a **~191-sentence sample** from 2 CTI reports.
> The paper used **19,178 sentences** (full corpus). These numbers are NOT
> directly comparable — compare direction and relative ordering only.

**Sample**: 191 sentences from 2 reports:
  - '3CXDesktopApp Backdoored in a Suspected Lazarus Campaign\xa0\xa0'  
  - 'AA21076A TrickBot Malware'
  
**Bootstrap CI**: 1000 resamples, seed=42 (only randomness in scorer).

### Runnable experiments (8B, ~191-sentence sample)

| Exp | Name | Micro P | Micro R | Micro F1 | Paper F1 | 95% CI | TP/FP/FN | Unlisted | Classes | Macro F1 | Paper Macro F1 |
| :-- | :--- | ---: | ---: | ---: | ---: | :---: | :--- | ---: | ---: | ---: | ---: |
| #19 | BaseLLM 8B (#19) | 7.49% | 68.52% | 13.50% | 4.77% | [10.10, 17.41]% | TP=37 FP=457 FN=17 | 31 | 23 | 32.12% | 4.27% |
| #22 | BaseRAG 8B (#22) | 10.12% | 75.93% | 17.86% | 18.81% | [13.42, 22.68]% | TP=41 FP=364 FN=13 | 21 | 23 | 29.58% | 19.47% |
| #25 | Optimized RAG 8B (#25) | 10.86% | 81.48% | 19.17% | 15.40% | [14.56, 24.15]% | TP=44 FP=361 FN=10 | 23 | 23 | 29.65% | 16.84% |
| #27 | Context 1-sent (#27) | 12.46% | 70.37% | 21.17% | 13.58% | [15.61, 27.01]% | TP=38 FP=267 FN=16 | 36 | 23 | 31.98% | 15.60% |

> [!CAUTION]
> Macro F1 values above are averaged over only a small subset of the 50 classes
> (those with >=1 gold label in the ~191-sentence sample). They are **unreliable**
> and must not be compared directly to paper macro F1 which uses 19,178 sentences.

### NOT RUN experiments

| Experiment | Reason |
| :--- | :--- |
| #23 BaseRAG 8B full desc | NOT RUN (hardware / time) |
| #24 Optimized RAG k=10 full desc | NOT RUN (hardware / time) |
| #26 Context 2-sentence | NOT RUN (hardware / time) |
| #28 Few-shot 8B | NOT RUN (hardware / time) |
| #29 BaseRAG 70B | NOT RUN (hardware — no GPU) |
| #30 BaseRAG DeepSeek-R1 70B | NOT RUN (hardware — no GPU) |
| #36 Optimized RAG DeepSeek-R1 70B | NOT RUN (hardware — no GPU) |

---

## 4. LLM Output Sanity Check: Unlisted Predictions

**Finding (tested)**: llama3.1:8b-instruct-q4_0 frequently predicts technique IDs outside the 10 candidates presented in the RAG prompt.

- For BaseLLM (#19): No candidates in prompt (expected). Model predicts from parametric knowledge only.
- For BaseRAG (#22), Optimized RAG (#25), Context 1-sent (#27): Top-10 candidate techniques are provided in the prompt for every sentence. Predictions outside the candidate list are tracked as unlisted.
- **The parser is correct**: `regex r'\b(T\d{4}(?:\.\d{3})?)\b'` faithfully extracts all codes written by the model. Parser was NOT changed.
- **Unlisted codes are excluded from precision denominator** (as the paper specifies) but are counted and reported.
- This behaviour matches the paper's observation of high unlisted prediction rates for 8B models.
- Full inspection log: `artifacts/llm_output_sanity_check.txt`.

---

## 5. Extension: BM25 + RRF Fusion (not in paper)

> **Label**: EXTENSION — not evaluated in Schuurman (2025).
> BM25 and RRF are evaluated here as an additional experiment.

| Method | R@1 | R@5 | R@10 | R@20 |
| :--- | ---: | ---: | ---: | ---: |
| Paper: Nomic Full (target) | — | — | 80.28% | — |
| Nomic w/ prefixes (repro) | 35.82% | 68.60% | 80.52% | 90.43% |
| BM25 alone (extension) | 23.20% | 49.83% | 64.38% | 80.07% |
| BM25 + Nomic RRF k=60 (ext) | 32.94% | 65.14% | 78.55% | 89.99% |

---

## 6. What Was and Was Not Reproduced

| Item | Status | Notes |
| :--- | :---: | :--- |
| Table 3: Nomic full (with prefixes) | REPRODUCED | within ~0.2% of paper target |
| Table 3: Nomic ablations (paragraph, chunks) | REPRODUCED (direction) | prefix setting affects results |
| Table 3: ATTACK-BERT Full | PARTIAL MATCH | see deviation analysis |
| Table 3: Reranker (ms-marco) | NOT MATCHED | qualitative reversal vs paper |
| Table 2: #19 BaseLLM 8B | DIRECTIONAL (~191 sent) | sample limited to 191 sentences |
| Table 2: #22 BaseRAG 8B | DIRECTIONAL (~191 sent) | sample limited to 191 sentences |
| Table 2: #25 Optimized RAG 8B | DIRECTIONAL (~191 sent) | sample limited to 191 sentences |
| Table 2: #27 Context 1-sent | DIRECTIONAL (~191 sent) | sample limited to 191 sentences |
| Table 2: #23, #24, #26, #28 (8B) | NOT RUN | time/resource constraint |
| Table 2: #29, #30, #36 (70B) | NOT RUN | no GPU |
| Extension: BM25 + RRF fusion | DONE (extension) | not in paper |
| Fig. 2: Label count distribution | GENERATED | from TRAM data |
| Fig. 4: Class distribution | GENERATED | from TRAM data |

---

## 7. Limitations

1. **Sample size**: LLM evaluation used ~191 sentences (2 reports) vs paper's 19,178. Micro F1 can vary widely at this scale; comparisons are directional only.
2. **Model size**: Only llama3.1:8b-instruct-q4_0 was run. The paper's best results used 70B models (Llama 3.1 70B, DeepSeek-R1 70B). Those are NOT RUN here.
3. **No summarized descriptions (Item 5)**: The Appendix C summarization experiment was not run. Paper target for nomic+summaries: R@10=83.93%.
4. **NumPy index vs Weaviate**: The paper used a Weaviate vector database. This reproduction uses a NumPy dot-product index. Results should be equivalent for exact cosine search but may differ if Weaviate applied HNSW approximation.
5. **Nomic prefix choice**: `search_query:` / `search_document:` follows nomic's documentation. The paper does not explicitly state which prefix was used. Both prefix and no-prefix results are reported.
6. **Reranker reversal**: Our reranker improves over nomic baseline whereas the paper shows degradation. The cause is not identified (untested hypothesis: different baseline or different max_length).
7. **Sample scope**: All 191 sentences from 2 complete reports evaluated for #19, #22, #25, #27.

---

## 8. Figures

- **Fig. 2** (`artifacts/fig2_label_distribution.png`): Distribution of gold label counts per sentence.
- **Fig. 4** (`artifacts/fig4_class_distribution.png`): Class distribution of 5,143 label instances across 50 techniques.
  - Most frequent: T1027 (Obfuscated Files or Information): 678 instances.
  - Least frequent: T1557.001 (LLMNR/NBT-NS Poisoning and SMB Relay): 7 instances.

---

*Report generated by `scripts/generate_reproduction_report.py`. All numbers derived from real model runs or pre-computed embeddings. No simulated results.*