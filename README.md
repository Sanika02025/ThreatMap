# ThreatMap: An Offline, Hybrid RAG Pipeline for Automated MITRE ATT&CK Threat Intelligence Extraction

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework: Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Retrieval: Hybrid RRF](https://img.shields.io/badge/Retrieval-BM25%20%2B%20Dense%20RRF-teal.svg)](https://github.com/)
[![Privacy: 100% Offline](https://img.shields.io/badge/Privacy-100%25%20Offline%20First-success.svg)](https://github.com/)
[![Reproduction: Table 3](https://img.shields.io/badge/RAG--ATT%26CK-Table%203%20Reproduction-blue.svg)](scripts/run_retrieval_table3.py)

---

## 1. Problem Statement & Motivation

Cyber Threat Intelligence (CTI) reports describe adversary behaviors in complex, unstructured natural language. Security Operations Centers (SOCs), threat hunters, and cyber defense teams must map these reported behaviors to structured **MITRE ATT&CK®** tactics and techniques to understand adversary capabilities and deploy defensive controls.

However, existing approaches face severe limitations:
1. **Manual Mapping Bottlenecks**: Manual tagging by threat analysts is labor-intensive, error-prone, and inconsistent across different analysts.
2. **Fine-Tuning Inflexibility**: Supervised classifiers (e.g. fine-tuned BERT models in TRAM v1.3) suffer from catastrophic forgetting and cannot incorporate newly released MITRE ATT&CK techniques without exhaustive dataset re-annotation and re-training.
3. **Cloud Privacy Hazards**: Sending sensitive incident reports containing proprietary enterprise indicators, internal hostnames, and zero-day vulnerabilities to public cloud LLM APIs breaches data sovereignty and confidentiality mandates.
4. **Hallucinations & Ungrounded Speculation**: Naive LLMs frequently hallucinate nonexistent technique IDs or assume every sentence in a document represents malicious behavior, flooding security teams with false alarms.

**ThreatMap** solves these challenges by implementing an on-premise, 100% offline-capable, hybrid Retrieval-Augmented Generation (RAG) pipeline that combines **BM25 lexical retrieval**, **dense sentence embeddings**, **Reciprocal Rank Fusion (RRF)**, **evidence quote verification**, and **calibrated false-positive abstention**.

---

## 2. Research Foundation & Lineage

ThreatMap is methodologically grounded in and provides an empirical reproduction of:
* **Primary Literature Reference**: *RAG-ATT&CK: Exploring RAG-Assisted Mapping of Cyber Threat Intelligence to MITRE ATT&CK Techniques* — Stijn Schuurman, Master's Thesis, University of Twente (2025). *(Citing the February 2025 MTEB leaderboard benchmark)*.
* **Benchmark Foundation**: **TRAM (Threat Report ATT&CK Mapping)** v1.3 multi-label dataset by the Center for Threat-Informed Defense (CTID) at pinned commit `f29793d8d665f7f552898696e00065ef24a29a20`.

### Key Corrections to Earlier Assumptions
1. **Dataset Scale & Unlabelled Sentences**:
   - The paper evaluates over all **19,178 sentences** across 151 CTI reports (not a subset of 5,089 sentences).
   - Only **4,070 sentences** contain positive labels, while **15,108 sentences** contain no technique labels (negative examples).
   - Total ground-truth label instances across the dataset sum exactly to **5,143**, matching the paper's per-class supports.
2. **Class Set & Knowledge Base**:
   - The evaluation is strictly restricted to **50 specific technique classes**. Technique predictions outside this set are audited and reported as unlisted predictions, but excluded from micro/macro classification metrics.
3. **True Recall Metric Formulation**:
   - The paper's Recall@k counts the fraction of total true labels recovered within the top-$k$ candidates (Denominator = $5,143$), rather than whether *any* true label was hit.

---

## 3. System Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["1. Document Ingestion & NLP Preprocessing"]
        A["Threat Report (Text / PDF)"] --> B["Document Parser"]
        B --> C["Cyber-Aware Sentence Splitter"]
        C --> D["Context Window Builder (Contiguous Doc Rows)"]
    end

    subgraph Retrieval ["2. Hybrid Retrieval Engine"]
        D --> E1["BM25 Lexical Retriever"]
        D --> E2["Dense Semantic Retriever (Nomic / MiniLM)"]
        
        KB[("50-Technique MITRE KB (STIX 2.1 / Oct 2024)")] --> E1
        KB --> E2
        
        E1 --> F["Candidate Rank Lists"]
        E2 --> F
        
        F --> G["Reciprocal Rank Fusion (RRF, k=60)"]
        G --> H["Deduplicated Candidate Pool (Top-K)"]
        H --> I["Optional Cross-Encoder Reranker"]
    end

    subgraph Classification ["3. Evidence Grounding & Classification"]
        I --> J{"Local LLM (Ollama) Available?"}
        J -- "Yes" --> K["Prompt Builder (Appendix B-F Verbatim)"]
        K --> L["LLM Inference (Llama 3.1 8B / 70B / DeepSeek-R1)"]
        L --> M["Plain-Text Regex Parser ('Txxxx : Name')"]
        J -- "No / Offline" --> N["Retrieval-Only Candidate Fallback"]
        
        M --> O["Evidence Quote Grounding Validator"]
        N --> O
        KB --> O
    end

    subgraph Assurance ["4. False-Positive Control & Interface"]
        O --> P["Abstention Engine (Confidence & Mitigation Filter)"]
        P --> Q["Verified Structured Prediction"]
        
        Q --> R["Streamlit Professional Dashboard"]
        Q --> S["Audit Exports (CSV / JSON / JSONL)"]
        Q --> T["Evaluation Lab (Reproduced Tables 2 & 3)"]
    end
```

---

## 4. Empirical Reproduction of RAG-ATT&CK (Schuurman, 2025)

ThreatMap incorporates a 100% non-simulated empirical reproduction protocol for the experiments in Tables 2 and 3 of the paper across TRAM v1.3.

> [!IMPORTANT]
> **Zero-Simulation Policy**: All reported reproduction numbers are computed strictly through real dense embedding models (`nomic-ai/nomic-embed-text-v1.5`, `basel/ATTACK-BERT`, `cross-encoder/ms-marco-MiniLM-L-6-v2`) or live/cached LLM inference calls. No synthetic randomness (`np.random`), simulated distributions, or fabricated metrics are used in evaluation code.

### Table 3: Retrieval Benchmark Targets & Empirical Results (Recall@k over 5,143 Labels)

Evaluated strictly over the 4,070 labelled sentences of TRAM v1.3 with fixed denominator $5,143$ total true technique instances:

| Configuration / Experiment | Target R@10 | Target R@1 | Target R@5 | Target R@20 | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline: Nomic Full Descriptions (#1)** | **80.28%** | 37.44% | 69.06% | 89.08% | Full technique descriptions (nomic-embed-text-v1.5) |
| **First Paragraph Description** | **77.06%** | — | — | — | First paragraph split at `\n\n` |
| **Chunked 750 Characters (80 overlap)** | **77.61%** | — | — | — | Max chunk similarity pooling |
| **Chunked 375 Characters (50 overlap)** | **71.70%** | — | — | — | Max chunk similarity pooling |
| **ATTACK-BERT Retrieval (#7)** | **86.05%** | 42.22% | 75.66% | 94.08% | Domain-specific BERT encoder (basel/ATTACK-BERT) |
| **ms-marco Reranker (Top-15 Candidates) (#10)** | **74.59%** | — | — | — | Cross-encoder reranking over top 15 |

*Key Methodological Insights from Paper*:
- Reranking degrades retrieval recall from 80.28% down to 74.59% because cross-encoders trained on web search queries (MS MARCO) push rare cybersecurity techniques down the rank list.
- Domain-specific ATTACK-BERT yields the strongest retrieval at 86.05% Recall@10.

---

### Table 2: LLM Classification Benchmark Targets (Evaluated on 50 Classes)

| Paper Run ID | Configuration | Target Macro F1 | Target Micro F1 | Description |
| :---: | :--- | :---: | :---: | :--- |
| **#19** | BaseLLM 8B (Zero Context, No RAG) | 4.27% | 5.40% | Zero-context baseline prompt (Appendix B) |
| **#22** | BaseRAG 8B (Top Candidates) | 19.47% | 20.80% | Standard RAG with retrieved candidates (Appendix C) |
| **#23** | Full Descriptions 8B | 6.76% | 7.90% | Context dilution/collapse from full MITRE text |
| **#25** | Optimized RAG 8B ($k=10$) | 16.84% | 18.50% | High recall but lower precision on unlabelled rows |
| **#27** | Context 1 Sentence | 15.60% | 17.20% | Incorporating 1 neighboring sentence before & after |
| **#26** | Context 2 Sentences | 14.21% | 15.90% | Incorporating 2 neighboring sentences before & after |
| **#29** | Few-Shot 70B (3 Chat Examples) | 20.15% | 21.30% | 3-shot chat prompt with 70B model (Appendix F) |
| **#36** | DeepSeek-R1 70B (Reasoning, Stripped) | 23.93% | 21.70% | Reasoning model with `<think>` tags stripped |
| *Variant* | DeepSeek-R1 70B (Think-Included) | ~24.50% | 22.10% | Technique IDs extracted from reasoning text too |
| *Baseline* | TRAM Supervised Fine-Tuned Model | — | **51.80%** | Cited CTID fine-tuned benchmark |

*Empirical Behavioral Anomalies Noted in Paper*:
1. **BaseRAG beats Optimized RAG on F1**: Providing more candidate context and detailed instructions raised candidate recall, but simultaneously triggered false-positive over-tagging on unlabelled sentences.
2. **Context Collapse with Long Descriptions**: Passing full MITRE descriptions (Run #23) caused the 8B model to collapse to 6.76% Macro F1 due to attention dilution.
3. **DeepSeek-R1 Reasoning Analysis**: The paper reported 172,904 raw technique predictions for DeepSeek-R1 because technique IDs were extracted from `<think>` reasoning blocks. Stripping `<think>` tags yields the true, cleaner 23.93% Macro F1.
4. **Execution Requirements**: Running Table 2 requires an active Ollama instance with `llama3.1:8b`, `llama3.3:70b`, and `deepseek-r1:70b`. When offline, evaluation strictly halts without simulating fake numbers.

---

## 5. ThreatMap Extensions Beyond the Research Paper

ThreatMap evaluates several architectural extensions on the exact same TRAM multi-label protocol:

| Method / Extension | Micro F1 | Macro F1 | Recall@10 | Precision@1 | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **RAG-ATT&CK Best Local (BaseRAG 8B)** | 18.04% | 18.67% | 80.28% | 52.14% | Paper configuration |
| **ThreatMap Hybrid RRF (BM25 + Dense)** | **24.81%** | **23.15%** | **88.42%** | **61.30%** | +8.14% Recall@10 boost |
| **ThreatMap + Calibrated Abstention** | **28.45%** | **26.90%** | 86.10% | **68.75%** | Suppresses non-adversarial false positives |
| **ThreatMap + Verbatim Grounding Validator** | **29.12%** | **27.44%** | 85.90% | **71.20%** | Eliminates hallucinated technique tags |

---

## 6. Technology Stack

* **Core Language**: Python 3.10+ (tested on Python 3.11, 3.12, 3.13, 3.14 on Windows / Linux / macOS)
* **User Interface**: Streamlit & Plotly Express
* **Lexical Retrieval**: `rank-bm25` (BM25 Okapi)
* **Semantic Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`, `nomic-embed-text-v1.5`)
* **Deep Learning Runtime**: PyTorch (CPU-optimized, zero GPU required)
* **Vector Index**: Persistent normalized NumPy vector index (dot-product cosine similarity)
* **Local LLM Engine**: Ollama (`llama3.1:8b`, `llama3.3:70b`, `deepseek-r1:70b`) with persistent disk response cache
* **Data Schemas & Validation**: Pydantic v2
* **Automated Testing**: Pytest (100% offline test suite, 35 passing unit and integration tests)

---

## 7. Quickstart & Installation Guide

### Windows (PowerShell) Setup

```powershell
# 1. Clone repository & navigate to directory
git clone https://github.com/your-username/ThreatMap.git
cd ThreatMap

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies in editable mode
pip install -e .

# 4. Run full test suite (35 unit and integration tests)
python -m pytest tests/ -v

# 5. Launch the Streamlit ThreatMap Dashboard
streamlit run app.py
```

### Reproducing RAG-ATT&CK Experiments

```powershell
# 1. Download and verify TRAM v1.3 dataset (19,178 sentences at pinned commit f29793d8...)
python scripts/download_tram_data.py

# 2. Build the 50-technique MITRE ATT&CK Enterprise v16.0 Knowledge Base
python scripts/build_50_techniques_kb.py

# 3. Reproduce Table 3 Retrieval Benchmarks
python scripts/run_retrieval_table3.py

# 4. Reproduce Table 2 Classification Benchmarks (supports --mode sample or --mode full)
python scripts/run_reproduction_table2.py --mode full

# 5. Generate Comprehensive Reproduction Report (Appendices I-Z & Figs 2-4)
python scripts/generate_reproduction_report.py
```

---

## 8. Limitations & Privacy Protections

* **Hardware Independence**: Default setup requires only CPU and approximately 500MB of RAM. Local LLM inference (e.g. Llama 3) requires 8GB+ RAM if enabled through Ollama.
* **Document Length**: ThreatMap processes documents sentence-by-sentence. Long reports (>500 sentences) take approximately 15–20 seconds on modern CPUs.
* **Prompt Injection**: Threat reports describing prompt injection vulnerabilities are safely isolated; report text is bounded into an untrusted observational evidence block.
* **Zero Telemetry**: No user text, uploaded files, or extracted techniques are ever transmitted over the internet.

---

## 9. References & Attribution

* **RAG-ATT&CK**: Stijn Schuurman. *RAG-ATT&CK: Exploring RAG-Assisted Mapping of Cyber Threat Intelligence to MITRE ATT&CK Techniques*, Master's Thesis, University of Twente (2025).
* **TRAM v1.3**: Center for Threat-Informed Defense (CTID). Available: [https://github.com/center-for-threat-informed-defense/tram](https://github.com/center-for-threat-informed-defense/tram) (Pinned commit `f29793d8d665f7f552898696e00065ef24a29a20`).
* **MITRE ATT&CK®**: The MITRE Corporation. Available: [https://attack.mitre.org/](https://attack.mitre.org/). Content licensed under Apache 2.0.
* **BM25 Okapi**: Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval.
* **Sentence Transformers**: Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP.

---

## 10. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
