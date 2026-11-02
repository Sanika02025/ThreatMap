# Limitations & Future Work

This document outlines the scope boundaries, unperformed experiments, empirical deviations, and future work for the RAG-ATT&CK reproduction study.

---

## 1. Limitations

1. **Sample Size for LLM Evaluation**:
   - The LLM classification evaluation was conducted on a sample of **2 complete CTI reports** comprising **191 sentences**, of which **49 sentences** contain in-scope gold labels (**54 gold label instances** across 23 technique classes).
   - The paper evaluated on all **19,178 sentences** from 151 reports. Because the sample size is small, bootstrap confidence intervals are wide (e.g., #22: [13.42, 22.68]%), and point estimates cannot be directly compared to full-corpus paper metrics.

2. **Model Scale & Hardware Constraints**:
   - Only `llama3.1:8b-instruct-q4_0` was evaluated on consumer CPU hardware (Intel i5-12450H, 16 GB RAM, no GPU).
   - The paper's primary and highest-performing configurations utilized **70B models** (Llama 3.1 70B and DeepSeek-R1 70B). These could not be run on this hardware and are marked `NOT RUN (hardware — no GPU)`.

3. **Unperformed Paper Experiments**:
   - **Summarized descriptions** (Paper Section 4.2 / Appendix C): LLM-generated summaries of technique descriptions (target R@10 = 83.93%) were not run.
   - **Few-shot prompting** (#28): Few-shot classification was not run due to context length and time constraints.
   - **Full-description prompt variants** (#23, #24): Providing full technique descriptions in the prompt was not run.
   - **2-sentence context window** (#26): Only 1-sentence context (#27) was evaluated; 2-sentence context was not run.

4. **Unperformed Embedding / Reranking Variants**:
   - **ATTACK-BERT CLS pooling**: Only mean pooling was evaluated (raw: 81.84%, cleaned: 82.09%). CLS pooling was not run due to hardware/time constraints.
   - **ms-marco Reranker with `max_length=512` and cleaned variants**: Only `max_length=256` with raw descriptions was evaluated (R@10 = 80.75%). `max_length=512` hung after model loading on CPU and was not run.

5. **No Separate Development Set**:
   - The evaluation was conducted without a separate tuning or development set. No hyperparameter optimization or tuning claims are made.

6. **Paper Ordering Inversion**:
   - The paper's reported ordering: `BaseRAG (#22: 18.81%) > Optimized (#25: 15.40%) > Context 1 (#27: 13.58%)` was **NOT reproduced** on the 191-sentence sample.
   - Our observed sample point estimates are `17.86% (#22) < 19.17% (#25) < 21.17% (#27)`. Because confidence intervals overlap heavily, no ordering among #22, #25, and #27 can be claimed from this sample.

7. **Unexplained Retrieval Deviations**:
   - **Reranker reversal**: In our run, the ms-marco reranker slightly exceeds the nomic baseline (80.75% vs 80.52%), whereas the paper reports that the reranker degraded recall (74.59% vs 80.28%). The cause is unidentified (untested hypotheses: candidate pool size, pair formatting, scoring thresholds, description cleaning).
   - **ATTACK-BERT gap**: Reproduced R@10 is 81.84% (raw) and 82.09% (cleaned), remaining 3.96–4.21 points below the paper's 86.05%. The gap remains unexplained (untested hypotheses: pooling method, tokenization/truncation differences, MITRE ATT&CK version discrepancies).

8. **Class Support and Retrieval Performance**:
   - Spearman rs = -0.07 (nomic) and -0.16 (ATTACK-BERT): no meaningful relationship between class support and Recall@10 was found.

---

## 2. Future Work

The following items are listed strictly as potential research avenues (no claims of expected outcome):

- Execution of full 19,178-sentence evaluation on GPU-enabled infrastructure.
- Evaluation of Llama 3.1 70B and DeepSeek-R1 70B models using quantized or hosted inference.
- Implementation of Appendix C description summarization and subsequent embedding evaluation.
- Systematic testing of CrossEncoder reranking parameters (max length 512, alternative top-k candidate pool sizes $k \in \{10, 15, 20\}$, text cleaning).
- Evaluation of ATTACK-BERT with CLS pooling and exact Hugging Face tokenizer truncation matching.
- Multi-dataset evaluation across disparate CTI benchmarks beyond TRAM v1.3.
- Investigation of hybrid retrieval fusion mechanisms with adaptive rank weighting.

