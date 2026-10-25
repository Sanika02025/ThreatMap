# ThreatMap Evaluation Protocol & Benchmarking Methodology

## 1. Purpose & Standards

This document establishes the experimental setup, evaluation metrics, and reproducible benchmarking protocol for ThreatMap. In strict adherence to scientific research standards:
* Metric formulas and zero-division conventions are documented explicitly.
* Benchmark datasets are preserved without synthetic label inflation.
* Ablation experiments are isolated and controlled.

---

## 2. Dataset Definition

The baseline evaluation uses the verified multilabel threat intelligence benchmark:
* **Source**: Synthesized from public APT threat advisories and aligned with the **TRAM v1.3** schema (Center for Threat-Informed Defense).
* **Sample Size**: 15 hand-verified evaluation sentences covering 11 distinct MITRE ATT&CK techniques across 8 tactics.
* **Negative / Abstention Cases**: 20% of sentences contain non-adversarial text (proactive threat hunting notes, defensive remediation, multi-factor authentication mandates) to rigorously measure false-positive control.

---

## 3. Metric Formulations

Let:
* $N$ be the number of test sentences.
* $C$ be the set of unique MITRE ATT&CK technique classes.
* $Y_i \subseteq C$ be the ground truth label set for sentence $i$.
* $\hat{Y}_i \subseteq C$ be the predicted label set for sentence $i$.
* $R_i = (c_{i,1}, c_{i,2}, \dots, c_{i,k})$ be the top-$k$ retrieved candidates for sentence $i$.

### Micro-Averaged Precision, Recall, and F1
Micro-metrics aggregate True Positives ($TP$), False Positives ($FP$), and False Negatives ($FN$) globally across all classes:

$$\text{Micro Precision} = \frac{\sum_{c \in C} TP_c}{\sum_{c \in C} (TP_c + FP_c)}$$

$$\text{Micro Recall} = \frac{\sum_{c \in C} TP_c}{\sum_{c \in C} (TP_c + FN_c)}$$

$$\text{Micro F1} = 2 \cdot \frac{\text{Micro Precision} \cdot \text{Micro Recall}}{\text{Micro Precision} + \text{Micro Recall}}$$

### Macro-Averaged Metrics
Macro-metrics compute unweighted arithmetic means across all classes, giving equal weight to frequent and rare techniques:

$$\text{Macro F1} = \frac{1}{|C|} \sum_{c \in C} F1_c$$

### Retrieval Recall@k
Measures whether the true technique was retrieved within the top-$k$ candidates:

$$\text{Recall}@k = \frac{1}{N_{\text{labeled}}} \sum_{i: |Y_i| > 0} \mathbb{I}\left(\{c_{i,1}, \dots, c_{i,k}\} \cap Y_i \neq \emptyset\right)$$

### Mean Reciprocal Rank (MRR)
Evaluates candidate ranking efficiency:

$$\text{MRR} = \frac{1}{N_{\text{labeled}}} \sum_{i: |Y_i| > 0} \frac{1}{\min \{r : c_{i,r} \in Y_i\} \cup \{\infty\}}$$

### Coverage & Abstention Rate
$$\text{Coverage} = \frac{N_{\text{predicted}}}{N}, \quad \text{Abstention Rate} = \frac{N_{\text{abstained}}}{N} = 1 - \text{Coverage}$$

---

## 4. Controlled Ablation Experiments Matrix

| Experiment | Method Name | Retrieval Mode | Candidate Top-K | Reranker | Context Strategy | Classifier |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **Exp A** | Lexical Retrieval Only | BM25 Okapi | 5 | Disabled | Target Only | Candidate Fallback |
| **Exp B** | Semantic Retrieval Only | Dense Vectors | 5 | Disabled | Target Only | Candidate Fallback |
| **Exp C** | Hybrid (RRF) | Reciprocal Rank Fusion | 5 | Disabled | Target Only | Candidate Fallback |
| **Exp D** | Hybrid (Weighted) | Normalized Score Fusion | 5 | Disabled | Target Only | Candidate Fallback |
| **Exp E** | Hybrid + Reranking | RRF + Cross-Encoder | 5 | Enabled | Target Only | Candidate Fallback |
| **Exp F** | Hybrid + Local LLM | RRF | 5 | Disabled | Target + Candidates | Ollama (Llama 3) |
| **Exp G** | Hybrid + Context LLM | RRF | 5 | Disabled | Context + Candidates | Ollama (Llama 3) |
| **Exp H** | Hybrid + Abstention | RRF (Threshold Calibrated) | 3 | Disabled | Target Only | Candidate Fallback |
