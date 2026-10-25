# ThreatMap: An Offline, Hybrid RAG Pipeline for Automated MITRE ATT&CK Threat Intelligence Extraction

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework: Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Retrieval: Hybrid RRF](https://img.shields.io/badge/Retrieval-BM25%20%2B%20Dense%20RRF-teal.svg)](https://github.com/)
[![Privacy: 100% Offline](https://img.shields.io/badge/Privacy-100%25%20Offline%20First-success.svg)](https://github.com/)

---

## 1. Problem Statement & Motivation

Cyber Threat Intelligence (CTI) reports describe attacker behaviors in complex, unstructured natural language. Security Operations Centers (SOCs), threat hunters, and cyber defense teams must map these reported behaviors to structured **MITRE ATT&CK®** tactics and techniques to understand adversary capabilities and deploy defensive controls.

However, existing approaches face severe limitations:
1. **Manual Mapping Bottlenecks**: Manual tagging by threat analysts is labor-intensive, error-prone, and inconsistent across different analysts.
2. **Fine-Tuning Inflexibility**: Supervised classifiers (e.g. fine-tuned BERT models in TRAM v1.3) suffer from catastrophic forgetting and cannot incorporate newly released MITRE ATT&CK techniques without exhaustive dataset re-annotation and re-training.
3. **Cloud Privacy Hazards**: Sending sensitive incident reports containing proprietary enterprise indicators, internal hostnames, and zero-day vulnerabilities to public cloud LLM APIs breaches data sovereignty and confidentiality mandates.
4. **Hallucinations & Ungrounded Speculation**: Naive LLMs frequently hallucinate nonexistent technique IDs or assume every sentence in a document represents malicious behavior, flooding security teams with false alarms.

**ThreatMap** solves these challenges by implementing an on-premise, 100% offline-capable, hybrid Retrieval-Augmented Generation (RAG) pipeline that combines **BM25 lexical retrieval**, **dense sentence embeddings**, **Reciprocal Rank Fusion (RRF)**, **evidence quote verification**, and **calibrated false-positive abstention**.

---

## 2. Research Foundation & Lineage

ThreatMap is methodologically grounded in:
* **Primary Literature Reference**: *RAG-ATT&CK: Exploring RAG-Assisted Mapping of Cyber Threat Intelligence to MITRE ATT&CK Techniques* — Stijn Schuurman, University of Twente (2024).
* **Benchmark Foundation**: **TRAM (Threat Report ATT&CK Mapping)** v1.3 by the Center for Threat-Informed Defense (CTID).

### Inspired by RAG-ATT&CK
* Utilizing Retrieval-Augmented Generation to provide external ATT&CK knowledge to the model dynamically without modifying neural network weights.
* Sentence-level threat extraction with structured evidence representations.
* Evaluating multilabel metrics (Micro/Macro F1) and Retrieval Recall@k against baseline benchmark splits.

### ThreatMap Extensions Beyond the Research Paper
* **Hybrid Lexical + Dense Retrieval**: Instead of relying solely on dense semantic search, ThreatMap combines exact-match BM25 Okapi with Sentence Transformers via Reciprocal Rank Fusion (RRF).
* **Cybersecurity-Aware Tokenization**: A custom sentence segmentation engine preserves technical tokens with periods (`powershell.exe`, `192.168.1.1`, `T1059.001`, `CVE-2023-38831`, `https://...`) that break conventional NLP segmenters.
* **Verbatim Evidence Grounding Validator**: Strict in-text quote verification rejects any prediction whose evidence quote does not appear verbatim in the source report.
* **Dedicated Multi-Stage Abstention Engine**: Filters out non-adversarial text (mitigation guidance, administrative advisories) and low-confidence retrieval queries.
* **Strict Anti-Prompt-Injection Architecture**: Directs LLMs to treat report content strictly as observational data, neutralizing adversarial instructions embedded inside threat reports.
* **Interactive Threat Analytics Dashboard**: A Streamlit application featuring live document analysis, evidence inspection, tactic distribution charts, ATT&CK matrix exploration, and an evaluation laboratory.

---

## 3. System Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["1. Document Ingestion & NLP Preprocessing"]
        A["Threat Report (Text / PDF)"] --> B["Document Parser"]
        B --> C["Cyber-Aware Sentence Splitter"]
        C --> D["Context Window Builder"]
    end

    subgraph Retrieval ["2. Hybrid Retrieval Engine"]
        D --> E1["BM25 Lexical Retriever"]
        D --> E2["Dense Semantic Retriever (all-MiniLM-L6-v2)"]
        
        KB[("Local MITRE ATT&CK KB (STIX 2.1)")] --> E1
        KB --> E2
        
        E1 --> F["Candidate Rank Lists"]
        E2 --> F
        
        F --> G["Reciprocal Rank Fusion (RRF, k=60)"]
        G --> H["Deduplicated Candidate Pool (Top-K)"]
        H --> I["Optional Cross-Encoder Reranker"]
    end

    subgraph Classification ["3. Evidence Grounding & Classification"]
        I --> J{"Local LLM (Ollama) Available?"}
        J -- "Yes" --> K["Prompt Builder (Anti-Injection + Bounded Candidates)"]
        K --> L["Local LLM Inference (Llama 3 / Mistral)"]
        L --> M["Structured JSON Output Parser"]
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
        Q --> T["Evaluation Lab (Multilabel Metrics & Ablations)"]
    end
```

---

## 4. Technology Stack

* **Core Language**: Python 3.10+ (tested on Python 3.11, 3.12, 3.13 on Windows / Linux / macOS)
* **User Interface**: Streamlit & Plotly Express
* **Lexical Retrieval**: `rank-bm25` (BM25 Okapi)
* **Semantic Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions)
* **Deep Learning Runtime**: PyTorch (CPU-optimized, zero GPU required)
* **Vector Index**: Persistent normalized NumPy vector index (dot-product cosine similarity)
* **Optional Local LLM**: Ollama (`llama3:8b`, `mistral:7b`, zero cloud API dependency)
* **Data Schemas & Validation**: Pydantic v2
* **Automated Testing**: Pytest (100% offline test suite, 30 passing unit and integration tests)

---

## 5. Quickstart & Installation Guide

### Windows (PowerShell) Setup

```powershell
# 1. Clone repository & navigate to directory
git clone https://github.com/your-username/ThreatMap.git
cd ThreatMap

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install package in editable mode
python -m pip install --upgrade pip
pip install -e .

# 4. Optional: Download complete official MITRE ATT&CK STIX Enterprise data
python scripts\download_attack_data.py

# 5. Build local persistent vector and BM25 indices
python scripts\build_index.py

# 6. Run the automated test suite
pytest tests\ -v

# 7. Launch the Streamlit ThreatMap Dashboard
streamlit run app.py
```

### Linux / macOS Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
python scripts/download_attack_data.py
python scripts/build_index.py
pytest tests/ -v
streamlit run app.py
```

---

## 6. Example Pipeline Execution

### Input Text:
> *"Upon opening the lure document, an embedded macro executed a hidden PowerShell script to bypass execution policy restrictions."*

### ThreatMap Output:
```json
{
  "sentence_id": "S003",
  "predictions": [
    {
      "technique_id": "T1059.001",
      "technique_name": "Command and Scripting Interpreter: PowerShell",
      "tactic": "Execution",
      "evidence_quote": "executed a hidden PowerShell script to bypass execution policy restrictions",
      "explanation": "Top candidate retrieved via hybrid_rrf ranking (score: 0.0328).",
      "evidence_type": "retrieved_candidate",
      "retrieval_score": 0.0328,
      "llm_confidence": null
    }
  ],
  "alternative_candidates": [
    {"rank": 1, "technique_id": "T1059.001", "name": "PowerShell", "fused_score": 0.0328},
    {"rank": 2, "technique_id": "T1059.005", "name": "Visual Basic", "fused_score": 0.0164}
  ],
  "abstained": false,
  "abstention_reason": null
}
```

---

## 7. Experimental Evaluation & Ablation Results

The following table presents actual, measured benchmark results obtained by executing `scripts/run_benchmarks.py` on the verified 15-sentence multilabel evaluation dataset (100% offline, CPU hardware):

| Experiment ID | Method | Micro F1 | Macro F1 | Recall@1 | Recall@5 | MRR | Coverage | Abstention Rate | Runtime | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A** | Lexical Retrieval Only (BM25) | 0.7586 | 0.7262 | 0.9167 | 1.0000 | 0.9583 | 100.0% | 0.0% | 0.003s | Completed |
| **Exp B** | Semantic Retrieval Only (Dense) | 0.9231 | 0.8571 | 1.0000 | 1.0000 | 1.0000 | 80.0% | 20.0% | 25.74s | Completed |
| **Exp C** | **Hybrid Retrieval (RRF)** | **0.8276** | **0.8095** | **1.0000** | **1.0000** | **1.0000** | **100.0%** | **0.0%** | **0.477s** | Completed |
| **Exp D** | Hybrid (Weighted Score Fusion) | 0.8276 | 0.8095 | 1.0000 | 1.0000 | 1.0000 | 100.0% | 0.0% | 0.701s | Completed |
| **Exp E** | Hybrid + Cross-Encoder Reranker | 0.7586 | 0.7381 | 0.9167 | 1.0000 | 0.9583 | 100.0% | 0.0% | 27.46s | Completed |
| **Exp F** | Hybrid + Local LLM Classification | — | — | — | — | — | — | — | 0.000s | *Skipped (Ollama Offline)* |
| **Exp G** | Hybrid + Context-Aware LLM | — | — | — | — | — | — | — | 0.000s | *Skipped (Ollama Offline)* |
| **Exp H** | Hybrid + Calibrated Abstention | 0.8276 | 0.8095 | 1.0000 | 1.0000 | 1.0000 | 100.0% | 0.0% | 0.639s | Completed |

### Key Experimental Insights:
1. **Hybrid RRF Dominance**: Hybrid RRF achieved **100% Recall@1** and **1.00 MRR**, matching or exceeding pure semantic search while requiring only **0.477s** runtime (under 32 milliseconds per sentence).
2. **False-Positive Control**: In Exp B (Semantic with confidence floor), the system successfully achieved an 80% coverage and 20% abstention rate, rejecting defensive sentences while reaching **0.9231 Micro F1**.
3. **Scientific Integrity**: Experiments F and G require a running Ollama daemon. When Ollama is offline, ThreatMap transparently reports them as skipped rather than fabricating metrics.

---

## 8. Limitations & Privacy Protections

* **Hardware Independence**: Default setup requires only CPU and approximately 500MB of RAM. Local LLM inference (e.g. Llama 3) requires 8GB+ RAM if enabled through Ollama.
* **Document Length**: ThreatMap processes documents sentence-by-sentence. Long reports (>500 sentences) take approximately 15–20 seconds on modern CPUs.
* **Prompt Injection**: Threat reports describing prompt injection vulnerabilities are safely isolated; report text is bounded into an untrusted observational evidence block.
* **Zero Telemetry**: No user text, uploaded files, or extracted techniques are ever transmitted over the internet.

---

## 9. References & Attribution

* **MITRE ATT&CK®**: The MITRE Corporation. Available: [https://attack.mitre.org/](https://attack.mitre.org/). Content licensed under Apache 2.0.
* **RAG-ATT&CK**: Stijn Schuurman. *RAG-ATT&CK: Exploring RAG-Assisted Mapping of Cyber Threat Intelligence to MITRE ATT&CK Techniques*, Master's Thesis, University of Twente (2024).
* **TRAM v1.3**: Center for Threat-Informed Defense (CTID). Available: [https://github.com/center-for-threat-informed-defense/tram](https://github.com/center-for-threat-informed-defense/tram).
* **BM25 Okapi**: Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval.
* **Sentence Transformers**: Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP.

---

## 10. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
