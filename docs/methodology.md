# ThreatMap Methodology & Mathematical Foundations

This document provides a beginner-friendly yet technically rigorous guide to the mathematical and conceptual foundations of ThreatMap.

---

## 1. The Threat Intelligence Challenge

Cyber threat intelligence (CTI) reports describe attacker tactics, techniques, and procedures (TTPs) using unstructured natural language. Security operations centers (SOCs) and threat hunters must map these descriptions to the **MITRE ATT&CK** matrix.

### Why Simple Keyword Search Fails
* **Vocabulary Mismatch**: A report might state *"credentials were harvested from LSASS memory"*, whereas the official technique title is *"OS Credential Dumping: LSASS Memory (T1003.001)"*.
* **Semantic Ambiguity**: The term *"command execution"* could refer to PowerShell (`T1059.001`), Windows Command Shell (`T1059.003`), or Unix Shell (`T1059.004`).
* **Noise and Mitigations**: Security advisories frequently contain non-adversarial text, such as *"Organizations must enforce MFA"*, which naive keyword matching flags as adversary activity.

---

## 2. Lexical Retrieval: BM25 Okapi

BM25 is a probabilistic information retrieval algorithm that evaluates matching keywords between a query and an indexed document.

For a query $Q$ containing terms $q_1, \dots, q_n$ and document $D$:

$$\text{Score}(D, Q) = \sum_{i=1}^n \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
* $f(q_i, D)$ is the term frequency of $q_i$ in document $D$.
* $|D|$ is the length of document $D$ in words.
* $\text{avgdl}$ is the average document length across the entire corpus.
* $k_1$ (typically $1.2 - 2.0$) calibrates term frequency saturation.
* $b$ (typically $0.75$) controls document length normalization.
* $\text{IDF}(q_i) = \ln \left( \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1 \right)$, penalizing ubiquitous words and emphasizing distinctive technical terms like `wevtutil` or `schtasks`.

---

## 3. Dense Semantic Retrieval & Cosine Similarity

While BM25 excels at exact keyword matching, semantic embeddings encode contextual meanings into high-dimensional geometric space.

### Embedding Generation
ThreatMap uses `sentence-transformers/all-MiniLM-L6-v2`, which maps sentences into $d = 384$ dimensional dense vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^{384}$.

### Cosine Similarity
The conceptual alignment between query vector $\mathbf{q}$ and technique vector $\mathbf{d}$ is calculated using the cosine of the angle between them:

$$\text{CosineSimilarity}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2} = \frac{\sum_{i=1}^d q_i d_i}{\sqrt{\sum_{i=1}^d q_i^2} \sqrt{\sum_{i=1}^d d_i^2}}$$

When all vectors are pre-normalized to unit length ($\|\mathbf{d}\|_2 = 1.0$), cosine similarity reduces to a fast dot product:

$$\text{CosineSimilarity}(\mathbf{q}, \mathbf{d}) = \mathbf{q} \cdot \mathbf{d}$$

---

## 4. Reciprocal Rank Fusion (RRF)

BM25 scores (unbounded positive values) and cosine similarity scores ($[-1.0, 1.0]$) operate on fundamentally disparate distributions. Normalizing and summing them directly often skews results toward whichever metric has higher variance.

To resolve this, ThreatMap uses **Reciprocal Rank Fusion (RRF)**:

$$\text{RRF}(d) = \sum_{m \in M} \frac{1}{k + \text{rank}_m(d)}$$

Where:
* $M$ is the set of retrieval systems ($M = \{\text{BM25}, \text{Semantic}\}$).
* $\text{rank}_m(d)$ is the 1-indexed position of document $d$ in system $m$.
* $k$ is a smoothing constant (standard default: $k = 60$).

**Why RRF Works**: Documents ranked highly by both systems receive exponentially higher scores than documents favored by only one system, providing robust consensus ranking without sensitive parameter tuning.

---

## 5. False-Positive Control & Abstention

A critical failure mode of standard RAG pipelines in cybersecurity is **hallucination of aggression**: assuming that every sentence in a report must describe an attack.

ThreatMap implements a multi-stage abstention engine:
1. **Semantic Similarity Floor**: Candidates below $s_{\min} = 0.30$ are rejected.
2. **Advisory Filter**: Sentences containing mitigation guidance or administrative notes are suppressed unless overwhelming adversary evidence exists.
3. **Evidence Grounding Verification**: Predicted techniques must be accompanied by verbatim quotes present in the report.
