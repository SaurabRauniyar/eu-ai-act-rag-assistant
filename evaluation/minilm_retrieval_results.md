# MiniLM Retrieval Evaluation

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`  
**Document chunks:** 160  
**Embedding dimensions:** 384  
**Retrieval:** FAISS cosine similarity  
**Top-K:** 5

## Results Summary

| Test | Query | Expected Article | Result | Status |
|---|---|---|---|---|
| 1 | Provider obligations | Article 16 | Rank 4 (0.7018) | 🟢 PASS |
| 2 | Human oversight | Article 14 | Rank 1 (0.8335) | 🟢 PASS |
| 3 | Deployer obligations | Article 26 | Rank 1 (0.8016) | 🟢 PASS |
| 4 | Prohibited AI practices | Article 5 | Not in Top 5 | 🔴 FAIL |

**Overall Top-5 retrieval:** 3/4 (75%)

---

## 🟢 Test 1 — Provider Obligations

**Question:**  
What are the obligations of providers of high-risk AI systems?

**Expected:**  
Article 16 — Obligations of providers of high-risk AI systems

**Retrieved:**  
Article 16 at **Rank 4**, similarity score **0.7018**.

**Status:** 🟢 **PASS**

---

## 🟢 Test 2 — Human Oversight

**Question:**  
What are the requirements for human oversight of high-risk AI systems?

**Expected:**  
Article 14 — Human oversight

**Retrieved:**  
Article 14 at **Rank 1**, similarity score **0.8335**.

**Status:** 🟢 **PASS**

---

## 🟢 Test 3 — Deployer Obligations

**Question:**  
What obligations do deployers of high-risk AI systems have?

**Expected:**  
Article 26 — Obligations of deployers of high-risk AI systems

**Retrieved:**  
Article 26 at **Rank 1**, similarity score **0.8016**.

**Status:** 🟢 **PASS**

---

## 🔴 Test 4 — Prohibited AI Practices

**Question:**  
What AI practices are prohibited under the EU AI Act?

**Expected:**  
Article 5 — Prohibited AI practices

**Retrieved:**  
Article 5 was **not retrieved in the Top 5**.

Top results:
1. Article 108 — 0.7492
2. Article 109 — 0.7480
3. Article 104 — 0.7362
4. Article 106 — 0.7331
5. Article 107 — 0.7243

**Status:** 🔴 **FAIL**




## Evaluation Summary

**🟢 Passed:** 3 / 4  
**🔴 Failed:** 1 / 4  
**Top-5 retrieval success on this test set:** **75%**

The failed query shows that MiniLM does not always rank the most relevant
legal provision highly. This provides a useful baseline for comparing
alternative embedding models such as BGE.



............................................
when checking 12 more
# MiniLM Retrieval Evaluation

**Model:** `sentence-transformers/all-MiniLM-L6-v2`  
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
**Retrieved:** Article 16 at **Rank 4**  
**Similarity:** `0.7018`

---

### 🟢 PASS — Human Oversight

**Question**  
> What are the requirements for human oversight of high-risk AI systems?

**Expected:** Article 14  
**Retrieved:** Article 14 at **Rank 1**  
**Similarity:** `0.8335`

---

### 🟢 PASS — Deployer Obligations

**Question**  
> What obligations do deployers of high-risk AI systems have?

**Expected:** Article 26  
**Retrieved:** Article 26 at **Rank 1**  
**Similarity:** `0.8016`

---

### 🔴 FAIL — Prohibited AI Practices

**Question**  
> What AI practices are prohibited under the EU AI Act?

**Expected:** Article 5  

**Result:** Article 5 was **not retrieved in the Top 5**.

---

### 🟢 PASS — Risk Management

**Question**  
> What risk management requirements apply to high-risk AI systems?

**Expected:** Article 9  
**Retrieved:** Article 9 at **Rank 1**  
**Similarity:** `0.7881`

---

### 🟢 PASS — Data Governance

**Question**  
> What are the data and data governance requirements for high-risk AI systems?

**Expected:** Article 10  
**Retrieved:** Article 10 at **Rank 1**  
**Similarity:** `0.7556`

---

### 🔴 FAIL — Technical Documentation

**Question**  
> What technical documentation is required for high-risk AI systems?

**Expected:** Article 11  

**Result:** Article 11 was **not retrieved in the Top 5**.

---

### 🟢 PASS — Record-Keeping

**Question**  
> What record-keeping and logging requirements apply to high-risk AI systems?

**Expected:** Article 12  
**Retrieved:** Article 12 at **Rank 1**  
**Similarity:** `0.8081`

---

### 🟢 PASS — Accuracy, Robustness and Cybersecurity

**Question**  
> What accuracy, robustness and cybersecurity requirements apply to high-risk AI systems?

**Expected:** Article 15  
**Retrieved:** Article 15 at **Rank 1**  
**Similarity:** `0.8109`

---

### 🟢 PASS — Transparency

**Question**  
> What transparency obligations apply to AI systems that interact with people or generate synthetic content?

**Expected:** Article 50  
**Retrieved:** Article 50 at **Rank 1**  
**Similarity:** `0.7228`

---

### 🟢 PASS — General-Purpose AI

**Question**  
> What are the obligations for providers of general-purpose AI models?

**Expected:** Article 53  
**Retrieved:** Article 53 at **Rank 1**  
**Similarity:** `0.7438`

---

### 🟢 PASS — Post-Market Monitoring

**Question**  
> What are the requirements for post-market monitoring of high-risk AI systems?

**Expected:** Article 72  
**Retrieved:** Article 72 at **Rank 1**  
**Similarity:** `0.7719`

---

## Evaluation Summary

| Metric | Result |
|---|---:|
| Questions | 12 |
| Passed Top-5 | 10 / 12 |
| Failed Top-5 | 2 / 12 |
| Hit@1 | **75% (9/12)** |
| Hit@3 | **75% (9/12)** |
| Hit@5 | **83.3% (10/12)** |

### Observations

MiniLM retrieved the expected article as the first result for 9 of the
12 evaluation questions.

However, it failed to retrieve Article 5 for the prohibited-practices
question and Article 11 for the technical-documentation question within
the Top 5.

These results provide the baseline for comparison with BGE.