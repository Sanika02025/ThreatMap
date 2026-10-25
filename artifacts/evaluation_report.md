# ThreatMap Quantitative Evaluation & Audit Report

**Generated:** 2026-10-19 09:22:27  
**Benchmark Dataset:** TRAM Sample Benchmark (15 examples)  
**Retrieval Method:** rrf (top_k=5)  
**Classification Mode:** Retrieval Candidate Fallback  

---

## 1. Global Multilabel Classification Metrics

| Metric | Score | Explanation |
| :--- | :---: | :--- |
| **Micro Precision** | `0.8000` | Global true positives divided by all predicted instances. |
| **Micro Recall** | `0.8571` | Global true positives divided by all ground truth labels. |
| **Micro F1** | `0.8276` | Harmonic mean of global precision and recall. |
| **Macro Precision** | `0.7857` | Unweighted mean precision across all evaluated technique classes. |
| **Macro Recall** | `0.8571` | Unweighted mean recall across all evaluated technique classes. |
| **Macro F1** | `0.8095` | Unweighted harmonic mean, sensitive to rare class performance. |

## 2. Information Retrieval Performance

| Retrieval Metric | Score | Research Interpretation |
| :--- | :---: | :--- |
| **Recall@1** | `1.0000` | Accuracy of the top-ranked technique candidate. |
| **Recall@3** | `1.0000` | Fraction of queries where ground truth is in top 3. |
| **Recall@5** | `1.0000` | Fraction of queries where ground truth is in top 5. |
| **Recall@10** | `1.0000` | Fraction of queries where ground truth is in top 10. |
| **Mean Reciprocal Rank (MRR)** | `1.0000` | Average reciprocal rank of the first relevant technique. |

## 3. Abstention & False-Positive Control

- **Coverage:** `100.0%` (15/15 sentences predicted)
- **Abstention Rate:** `0.0%` (safely abstained on ambiguous or non-adversarial sentences)
- **Total Latency:** `0.571 s` (38.1 ms/sentence)

## 4. Error Analysis Summary

- **False Positives:** 3
- **False Negatives:** 2
- **Total Abstentions:** 0

### Sample False Positives (Over-predictions):
- **Text**: "Security analysts conducted proactive threat hunting across corporate endpoints during Q3 2024."
  - Predicted: `T1003.001` (OS Credential Dumping: LSASS Memory)
  - Ground Truth: `[]`
- **Text**: "All enterprise administrators must enforce multi-factor authentication on every remote portal."
  - Predicted: `T1021.001` (Remote Services: Remote Desktop Protocol)
  - Ground Truth: `[]`
- **Text**: "Incident response personnel successfully recovered affected domain controllers from immutable offline backups."
  - Predicted: `T1105` (Ingress Tool Transfer)
  - Ground Truth: `[]`

### Sample False Negatives (Missed Detections):
- **Text**: "The malicious document executed a stealthy PowerShell command to download subsequent staging files."
  - Missed Technique: `T1105`
  - Retrieval Status: Not in top candidates
- **Text**: "The intrusion operators ran whoami and net user to discover local user accounts and domain privileges."
  - Missed Technique: `T1087.001`
  - Retrieval Status: Rank 2
