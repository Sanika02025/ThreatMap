# Reproduction Guide: Environment, Pipeline & Timings

This document provides exact reproduction instructions, execution order, software environment specifications, hardware details, and empirical wall-clock timings recorded across artifacts.

---

## 1. System & Environment Specifications

All specifications below were read directly from the runtime environment via `platform`, `sys`, and `importlib.metadata`:

| Property | Value |
| :--- | :--- |
| **Operating System** | Windows 11 (10.0.26200-SP0) |
| **Processor** | Intel Core i5-12450H (Intel64 Family 6 Model 154 Stepping 3, GenuineIntel) |
| **RAM** | 16 GB |
| **GPU** | None (CPU only) |
| **Python Version** | 3.14.0 (`tags/v3.14.0:ebf955d`, MSC v.1944 64 bit AMD64) |
| **Repository Git Commit** | `5e273e5fe9c2dcf15b9745378fb28fbfacc2ba3e` |
| **TRAM Pinned Commit** | `f29793d8d665f7f552898696e00065ef24a29a20` |
| **Ollama Model Tag** | `llama3.1:8b-instruct-q4_0` |
| **Ollama Model Digest** | unknown (ollama CLI not in PATH) |
| **Bootstrap Random Seed** | `42` (`numpy.random.default_rng(seed=42)`, 1,000 iterations — sole randomness in scorer) |

### Installed Package Versions (`importlib.metadata`)

| Package | Version |
| :--- | :--- |
| `sentence-transformers` | `6.2.0` |
| `torch` | `2.15.0` |
| `numpy` | `2.5.3` |
| `rank-bm25` | `0.2.2` |
| `scikit-learn` | `1.9.1` |
| `pandas` | `2.3.3` |

---

## 2. Execution Order to Reproduce All Artifacts

Execute the following commands sequentially from the repository root:

### Step 1: Run Retrieval Evaluation (Table 3)
```powershell
python scripts/run_retrieval_table3.py
```
- **Inputs**: `data/tram2-data/multi_label.json`, `data/processed/mitre_attack_50_techniques.json`, `artifacts/embedding_cache/*.npy`
- **Output**: `artifacts/retrieval_table3_real_results.json`

### Step 2: Run ATTACK-BERT & Reranker Ablations
```powershell
python scripts/investigate_table3_misses.py
python scripts/build_ablation_results.py
```
- **Inputs**: `artifacts/embedding_cache/*.npy`
- **Output**: `artifacts/table3_ablation_investigation.json`

### Step 3: Run BM25 and RRF Extension
```powershell
python scripts/run_bm25_rrf.py
```
- **Inputs**: `data/tram2-data/multi_label.json`, `data/processed/mitre_attack_50_techniques.json`, `artifacts/embedding_cache/nomic_*.npy`
- **Output**: `artifacts/bm25_rrf_results.json`

### Step 4: Score LLM Sample (Table 2)
```powershell
python scripts/score_llm_sample.py --mode sample --experiments 19 22 25 27
```
- **Inputs**: `artifacts/llm_cache/exp*.jsonl`, `data/tram2-data/multi_label.json`, `data/processed/mitre_attack_50_techniques.json`
- **Output**: `artifacts/llm_sample_scored_report.json`

### Step 5: Run Retrieval Ceiling & Error Decomposition Analysis
```powershell
python scripts/analyze_retrieval_ceiling.py
```
- **Inputs**: `artifacts/llm_cache/exp*.jsonl`, `artifacts/embedding_cache/sample_191_*.npy`, `data/tram2-data/multi_label.json`
- **Output**: `artifacts/retrieval_ceiling.json`

### Step 6: Run Per-Class Retrieval Analysis
```powershell
python scripts/analyze_per_class.py
```
- **Inputs**: `artifacts/embedding_cache/*.npy`, `data/tram2-data/multi_label.json`
- **Outputs**: `artifacts/per_class_retrieval.json`, `artifacts/per_class_retrieval.md`

### Step 7: Plot Recall@k Curves (k = 1..50)
```powershell
python scripts/plot_recall_curve.py
```
- **Inputs**: `artifacts/embedding_cache/*.npy`, `data/tram2-data/multi_label.json`
- **Outputs**: `artifacts/fig_recall_at_k.png`, `artifacts/recall_at_k.json`

### Step 8: Generate Figures 2 & 4
```powershell
python scripts/generate_figures.py
```
- **Inputs**: `data/tram2-data/multi_label.json`
- **Outputs**: `artifacts/fig2_label_distribution.png`, `artifacts/fig4_class_distribution.png`

### Step 9: Generate Master Reproduction Report
```powershell
python scripts/generate_reproduction_report.py
```
- **Inputs**: `artifacts/retrieval_table3_real_results.json`, `artifacts/llm_sample_scored_report.json`, `artifacts/bm25_rrf_results.json`, `artifacts/retrieval_ceiling.json`, `data/tram2-data/multi_label.json`
- **Output**: `artifacts/reproduction_report.md`

---

## 3. Empirical Wall-Clock Timings

### Retrieval & Ranking Timings (from `artifacts/retrieval_table3_real_results.json` & `artifacts/bm25_rrf_results.json`)

| Task / Configuration | Wall-Clock Time (s) |
| :--- | ---: |
| Nomic Full (with prefixes) ranking | 0.10 s |
| Nomic First Paragraph (with prefixes) ranking | 0.10 s |
| Nomic Chunk 750 (with prefixes) ranking | 0.10 s |
| Nomic Chunk 375 (with prefixes) ranking | 0.10 s |
| Nomic Full (no prefixes) ranking | 0.08 s |
| Nomic First Paragraph (no prefixes) ranking | 0.11 s |
| Nomic Chunk 750 (no prefixes) ranking | 0.13 s |
| Nomic Chunk 375 (no prefixes) ranking | 0.12 s |
| ATTACK-BERT Full ranking | 0.71 s |
| ms-marco Reranker (Top 15) ranking | 0.40 s |
| BM25 index build (50 descriptions) | 0.04 s |
| BM25 scoring (4,070 queries) | 9.42 s |
| BM25 + Nomic RRF fusion (k=60) | 0.17 s |

### LLM Inference Timings (recorded in `artifacts/llm_cache/*.jsonl`)

Evaluated on 191 sentences per experiment using Ollama `llama3.1:8b-instruct-q4_0` on CPU:

| Experiment | Total Duration (s) | Total Duration (h) | Mean per Sentence (s) |
| :--- | ---: | ---: | ---: |
| #19 BaseLLM 8B | 9,482.5 s | 2.63 h | 49.65 s |
| #22 BaseRAG 8B | 4,372.4 s | 1.21 h | 22.89 s |
| #25 Optimized RAG 8B | 4,175.7 s | 1.16 h | 21.86 s |
| #27 Context 1-sentence | 5,413.6 s | 1.50 h | 28.34 s |

