# ThreatMap Viva Voce & Technical Interview Guide

This guide prepares third-year Computer Engineering students for academic vivas, project defenses, and technical interview questions regarding ThreatMap.

---

### Q1: Why is ThreatMap considered an NLP and Information Retrieval project rather than just a cybersecurity tool?
**Answer**:
ThreatMap addresses a core Natural Language Processing challenge: extracting structured semantics and behavioral intent from unstructured, heterogeneous natural language text. It solves vocabulary mismatch, multi-sentence contextual dependencies, coreference resolution, and multilabel classification using Information Retrieval (IR) algorithms including BM25 Okapi, dense semantic representations via transformer sentence embeddings, and Reciprocal Rank Fusion (RRF).

---

### Q2: Why is keyword matching alone insufficient for mapping threat intelligence to MITRE ATT&CK?
**Answer**:
Adversary reports describe behaviors using varied terminology and vernacular that rarely match official technique names verbatim. For instance, a report stating *"the intrusion set executed a macro that spawned a hidden command shell"* describes technique `T1059.005` (Visual Basic) and `T1059.003` (Windows Command Shell), yet neither phrase matches the text verbatim. Conversely, benign words like *"powershell"* can appear in hardening guidelines. ThreatMap solves this by fusing dense contextual embeddings with BM25.

---

### Q3: What is RAG, and why is it preferred over fine-tuning a BERT model for this task?
**Answer**:
Retrieval-Augmented Generation (RAG) decouples the knowledge base from model weights. In cybersecurity:
1. **Knowledge Agility**: The MITRE ATT&CK enterprise matrix is continually updated with new techniques, sub-techniques, and platform descriptions. A fine-tuned classifier (like TRAM v1.3) requires complete dataset re-annotation and re-training for every ATT&CK update. With RAG, updating the knowledge base merely requires re-indexing.
2. **Explainability & Grounding**: RAG provides retrieved reference documents that directly justify each classification.
3. **Hardware Efficiency**: RAG operates efficiently on CPU without requiring GPU training pipelines.

---

### Q4: How does Reciprocal Rank Fusion (RRF) work, and why not just add BM25 and Cosine Similarity scores together?
**Answer**:
BM25 scores and Cosine Similarities have fundamentally incompatible score distributions: BM25 produces unbounded positive values dependent on document lengths, while Cosine Similarity produces values in $[-1, 1]$ (or $[0, 1]$ after normalization). Adding them directly causes arbitrary dominance by whichever metric has greater variance.
RRF bypasses score calibration by operating purely on the rank order:
$$\text{RRF}(d) = \sum_{m \in M} \frac{1}{k + \text{rank}_m(d)}$$
With smoothing parameter $k = 60$, a candidate ranked near the top in both modalities receives an exponentially higher score, achieving high recall without brittle score threshold tuning.

---

### Q5: How does ThreatMap prevent AI hallucinations?
**Answer**:
1. **Dynamic Knowledge Base Bounding**: The classifier's predictions are strictly validated against a local STIX 2.1 knowledge base. Any technique ID not found in the official matrix is immediately rejected.
2. **Verbatim Evidence Verification**: The system verifies that the evidence quote provided by the model exists verbatim within the original report sentence, rejecting ungrounded claims.
3. **Multi-Stage Abstention**: When similarity scores fall below threshold or when the text describes non-adversarial mitigations, ThreatMap abstains rather than forcing a speculative prediction.

---

### Q6: What is the difference between Micro F1 and Macro F1 in multilabel evaluation?
**Answer**:
* **Micro F1**: Aggregates True Positives, False Positives, and False Negatives globally across all sentences. It reflects overall instance-level accuracy but is heavily dominated by frequent techniques (e.g. `T1059.001`).
* **Macro F1**: Calculates the F1 score for each technique class independently and computes their unweighted mean. It gives equal importance to rare techniques, making it sensitive to class imbalance and long-tail performance.

---

### Q7: Why is 100% offline capability a critical engineering requirement for ThreatMap?
**Answer**:
Threat intelligence reports frequently contain sensitive proprietary incident data, internal IP addresses, zero-day vulnerabilities, or confidential customer telemetry. Transmitting this data to public cloud LLMs (such as OpenAI or Anthropic APIs) violates privacy policies and data residency laws. ThreatMap runs local sentence embedding inference and local Ollama inference entirely on-premise.
