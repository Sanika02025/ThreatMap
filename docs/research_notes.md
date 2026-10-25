# ThreatMap Research Notes & Related Work

## 1. Literature Review & Foundation

### RAG-ATT&CK (Stijn Schuurman, University of Twente, 2024)
* **Title**: *RAG-ATT&CK: Exploring RAG-Assisted Mapping of Cyber Threat Intelligence to MITRE ATT&CK Techniques*
* **Core Investigation**: Evaluated whether Retrieval-Augmented Generation (RAG) could bypass the costly retraining requirement of fine-tuned language models (such as TRAM and RoBERTa) when mapping CTI sentences to MITRE ATT&CK techniques.
* **Findings**:
  * Dense semantic retrieval consistently outperformed pure lexical retrieval on complex, descriptive sentences.
  * Incorporating technique descriptions into LLM prompts yielded significant recall improvements over zero-context LLM classification.
  * In the author's experiments, RAG-ATT&CK achieved competitive performance with TRAM v1.3 on several metrics, though it did not strictly surpass fine-tuning on every single metric.
  * A crucial finding was knowledge agility: RAG seamlessly accommodates newly updated or versioned ATT&CK techniques without retraining neural network weights.

### TRAM (Threat Report ATT&CK Mapping) — CTID
* Built by the Center for Threat-Informed Defense (CTID).
* Employs fine-tuned transformer architectures (e.g., SciBERT, RoBERTa) trained on human-annotated CTI sentences.
* Limitation: Fine-tuned classifiers suffer from catastrophic forgetting and cannot adapt to updated ATT&CK matrix releases without re-annotation and retraining.

---

## 2. ThreatMap's Contributions Beyond Prior Work

| Dimension | RAG-ATT&CK (Schuurman, 2024) | ThreatMap (Our Implementation) |
| :--- | :--- | :--- |
| **Retrieval Strategy** | Dense Semantic Search only | **Hybrid Reciprocal Rank Fusion (BM25 + Dense Embeddings)** |
| **False-Positive Control** | Assumed adversarial context | **Dedicated Multi-Stage Abstention Engine** (Mitigation & Confidence Filters) |
| **Evidence Grounding** | LLM-generated explanations | **Verbatim Evidence Quote Extraction & In-Text Verification** |
| **Hallucination Defense** | Prompt-only constraints | **Strict Dynamic Knowledge Base Validation Filter** |
| **Execution Privacy** | Cloud API / External LLMs | **100% Offline-Capable (Local Embeddings + Optional Ollama)** |
| **Preprocessing** | Standard sentence splitting | **Cybersecurity-Aware Tokenizer** (Preserves IPs, CVEs, file extensions) |

---

## 3. Core Research Questions (RQ1 - RQ4)

* **RQ1 (Hybrid Retrieval)**: *How does hybrid lexical-semantic retrieval compare with lexical-only and semantic-only retrieval for identifying relevant MITRE ATT&CK candidates?*
  * *Investigation*: Measures Retrieval Recall@k and MRR across BM25, dense embeddings, and RRF.
* **RQ2 (Context-Awareness)**: *Does incorporating neighboring sentence context improve technique mapping quality over target-sentence-only classification?*
  * *Investigation*: Tests whether adjacent sentences resolve pronoun anaphora and multi-step attack narratives.
* **RQ3 (Abstention Trade-off)**: *How does confidence-based abstention impact prediction precision, coverage, and error rates on non-adversarial sentences?*
  * *Investigation*: Evaluates false-positive suppression on administrative and mitigation statements.
* **RQ4 (Resource Efficiency)**: *What are the latency, memory, and throughput trade-offs of an offline-first RAG pipeline on commodity CPU hardware?*
  * *Investigation*: Quantifies sub-100ms inference on commodity student hardware without GPU requirements.
