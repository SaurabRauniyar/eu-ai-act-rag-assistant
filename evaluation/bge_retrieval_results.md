# BGE Retrieval Evaluation

**Model:** `BAAI/bge-small-en-v1.5`  
**Dataset:** EU AI Act  
**Chunks:** 160  
**Embedding dimension:** 384  
**Vector search:** FAISS + cosine similarity  
**Top-K:** 5  
**Evaluation questions:** 12  

---

## Evaluation Results

### 🟢 PASS — Provider Obligations

**Question**  
> What are the obligations of providers of high-risk AI systems?

**Expected:** Article 16  
**Retrieved:** Article 16 at **Rank 2**  
**Similarity:** `0.8402`

---

### 🟢 PASS — Human Oversight

**Question**  
> What are the requirements for human oversight of high-risk AI systems?

**Expected:** Article 14  
**Retrieved:** Article 14 at **Rank 1**  
**Similarity:** `0.8618`

---

### 🟢 PASS — Deployer Obligations

**Question**  
> What obligations do deployers of high-risk AI systems have?

**Expected:** Article 26  
**Retrieved:** Article 26 at **Rank 1**  
**Similarity:** `0.8655`

---

### 🟢 PASS — Prohibited AI Practices

**Question**  
> What AI practices are prohibited under the EU AI Act?

**Expected:** Article 5  
**Retrieved:** Article 5 at **Rank 1**  
**Similarity:** `0.8046`

---

### 🟢 PASS — Risk Management

**Question**  
> What risk management requirements apply to high-risk AI systems?

**Expected:** Article 9  
**Retrieved:** Article 9 at **Rank 3**  
**Similarity:** `0.8067`

---

### 🟢 PASS — Data Governance

**Question**  
> What are the data and data governance requirements for high-risk AI systems?

**Expected:** Article 10  
**Retrieved:** Article 10 at **Rank 1**  
**Similarity:** `0.8533`

---

### 🟢 PASS — Technical Documentation

**Question**  
> What technical documentation is required for high-risk AI systems?

**Expected:** Article 11  
**Retrieved:** Article 11 at **Rank 2**  
**Similarity:** `0.8056`

---

### 🟢 PASS — Record-Keeping

**Question**  
> What record-keeping and logging requirements apply to high-risk AI systems?

**Expected:** Article 12  
**Retrieved:** Article 12 at **Rank 1**  
**Similarity:** `0.8602`

---

### 🟢 PASS — Accuracy, Robustness and Cybersecurity

**Question**  
> What accuracy, robustness and cybersecurity requirements apply to high-risk AI systems?

**Expected:** Article 15  
**Retrieved:** Article 15 at **Rank 1**  
**Similarity:** `0.8835`

---

### 🟢 PASS — Transparency

**Question**  
> What transparency obligations apply to AI systems that interact with people or generate synthetic content?

**Expected:** Article 50  
**Retrieved:** Article 50 at **Rank 1**  
**Similarity:** `0.8434`

---

### 🟢 PASS — General-Purpose AI

**Question**  
> What are the obligations for providers of general-purpose AI models?

**Expected:** Article 53  
**Retrieved:** Article 53 at **Rank 1**  
**Similarity:** `0.8786`

---

### 🟢 PASS — Post-Market Monitoring

**Question**  
> What are the requirements for post-market monitoring of high-risk AI systems?

**Expected:** Article 72  
**Retrieved:** Article 72 at **Rank 1**  
**Similarity:** `0.8289`

---

## Evaluation Summary

| Metric | Result |
|---|---:|
| Questions | 12 |
| Passed Top-5 | 12 / 12 |
| Failed Top-5 | 0 / 12 |
| Hit@1 | **75% (9/12)** |
| Hit@3 | **100% (12/12)** |
| Hit@5 | **100% (12/12)** |

### Observations

BGE retrieved the expected EU AI Act article within the Top 5 for all
12 evaluation questions.

For 9 of the 12 questions, the expected article was the first retrieved
result. The remaining expected articles appeared at Rank 2 or Rank 3.

The results show strong retrieval performance on this small,
manually constructed evaluation set. They should not be interpreted
as general accuracy of the model.

The next experiment compares these results with the MiniLM baseline
using the same chunks, questions, FAISS search configuration and
Top-K value.