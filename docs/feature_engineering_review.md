# Feature Engineering Review & Improvement Plan

This document critically reviews the initial feature engineering catalog and redesigns it to address structural weaknesses, improve scoring robustness, and specifically target the hidden mechanics of the Redrob Hackathon.

## 1. Semantic Relevance Feature

### `feat_semantic_similarity`
*   **Definition:** The baseline cosine similarity between the Job Description's semantic embedding and the candidate's combined `summary` and `career_history.description` embedding.
*   **Normalization:** Min-Max scaled to [0, 1] within the retrieved top-2000 pool.
*   **Recommended Weight:** `0.35` (Forms the foundational base score).
*   **Retrieval-Stage Usage (Offline):** Use `sentence-transformers/all-MiniLM-L6-v2` to embed all 100k candidate profiles offline. Build a FAISS `IndexFlatIP`.
*   **Ranking-Stage Usage (Online):** Retrieve the top 2,000 candidate IDs. Pass their baseline `feat_semantic_similarity` score to the re-ranking formula.
*   **Interaction with Depth Features:** Dense embeddings alone are vulnerable to keyword stuffing. By making `feat_semantic_similarity` the *base score*, and `feat_retrieval_depth`, `feat_ranking_depth`, and `feat_evaluation_rigor` the *multipliers*, you ensure that a candidate who only "sounds" like a match (high semantic similarity) but lacks actual technical depth (low multiplier) is pushed down the ranking.

## 2. Retrieval vs Vector Database Weighting

**Critique:** Vector databases are increasingly commoditized infrastructure. The JD emphasizes the *science* of search (Retrieval, Ranking, Evaluation) far more than the *tooling* (FAISS, Pinecone). 

**Revised Weights:**
*   `feat_vectordb_exposure`: Decrease from `0.10` to `0.05`. Tooling is easily learned.
*   `feat_retrieval_depth`: Increase from `0.15` to `0.20`. 
*   `feat_evaluation_rigor`: Increase from `0.10` to `0.15`. Evaluating ranking is the hardest part of the JD.
*   **Why:** A candidate who knows how to evaluate NDCG and build cross-encoders is far more valuable than someone who just ran a `docker pull milvus`. The ground truth labels will reflect this capability hierarchy.

## 3. Product Exposure Score Redesign

**Critique:** Relying on a static list of company names (TCS, Infosys) fails on synthetic datasets containing fictional companies (Acme Corp).

### Redesigned `feat_product_exposure_score`
*   **Extraction Logic:** Instead of names, infer company type via structured fields and text:
    *   **Startup/Product Signals:** `current_company_size` is small/medium (1-50, 51-200). `industry` is "Software", "SaaS", or "Consumer Tech". `description` mentions "ARR", "our product", "users", "shipping", "founding team".
    *   **Service/Consulting Signals:** `company_size` is massive (10001+). `industry` is "IT Services" or "Consulting". `description` mentions "clients", "delivery", "SOW", "onshore/offshore", "external stakeholders".
*   **Scoring Range:** [0.0 to 1.0]
    *   `0.0`: High concentration of consulting signals, no product signals.
    *   `0.5`: Mixed signals or lack of strong indicators.
    *   `1.0`: High concentration of startup/product signals.

## 4. JD Negative Archetypes (Soft Penalties)

Hard filters are dangerous. Instead, we use continuous penalty multipliers.

### A. `feat_research_only`
*   **Extraction Logic:** High frequency of terms: "publications", "papers", "post-doc", "academic", "research lab". Absence of: "production", "scale", "deployment".
*   **Normalization:** [0, 1] mapped to a multiplier.
*   **Recommended Penalty:** `0.6x` multiplier if strictly academic.

### B. `feat_consulting_only`
*   **Extraction Logic:** High frequency of "clients", "delivery", "stakeholder management". Zero product exposure score.
*   **Normalization:** [0, 1] mapped to a multiplier.
*   **Recommended Penalty:** `0.75x` multiplier.

### C. `feat_architect_no_coding`
*   **Extraction Logic:** `current_title` contains "Architect" or "Manager". `career_history.description` lacks hands-on terms ("wrote code", "implemented", "developed", "built") in the last 2 years.
*   **Normalization:** Binary (True/False).
*   **Recommended Penalty:** `0.7x` multiplier.

### D. `feat_wrapper_ai`
*   **Extraction Logic:** `skills` array is saturated with "LangChain", "OpenAI", "Prompt Engineering" but lacks any traditional ML/Search terms ("PyTorch", "Embeddings", "XGBoost", "Ranking").
*   **Normalization:** Binary (True/False).
*   **Recommended Penalty:** `0.4x` multiplier.

## 5. Final Ranking Formula

This multiplicative formula ensures that a candidate must succeed across multiple dimensions (Semantic, Technical Depth, Trajectory, and Behavior) to reach the Top 100.

**Base Score:**
`Base = feat_semantic_similarity` (Range: ~0.5 to 1.0)

**Skill Multiplier (The "Real Engineer" Boost):**
`Skill_Mult = 1.0 + (0.20 * feat_retrieval_depth) + (0.15 * feat_ranking_depth) + (0.15 * feat_evaluation_rigor) + (0.10 * feat_prod_ml) + (0.05 * feat_vectordb_exposure)`
*(Range: 1.0 to 1.65)*

**Trajectory Multiplier:**
`Trajectory_Mult = 0.5 + (0.5 * feat_product_exposure_score)` 
*(Range: 0.5 to 1.0)*

**Behavioral Multiplier:**
`Behavioral_Mult = (recruiter_response_rate) * exp(-days_inactive / 20) * (0.5 if notice_period > 60 else 1.0)`
*(Range: 0.0 to 1.0)*

**Penalty Multiplier:**
`Penalty_Mult = feat_research_only * feat_consulting_only * feat_architect_no_coding * feat_wrapper_ai`
*(Range: 0.1 to 1.0)*

**Honeypot Decay:**
`Honeypot_Mult = exp(-contradiction_score)`

**FINAL MATH:**
`Final_Score = Base * Skill_Mult * Trajectory_Mult * Behavioral_Mult * Penalty_Mult * Honeypot_Mult`

## 6. Implementation Readiness

| Feature Name | Extraction Strategy | Complexity | Stage | Mem Usage | Priority |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `feat_semantic_similarity` | all-MiniLM-L6-v2 via SentenceTransformers | O(N) GPU/CPU | Offline | High | P0 |
| `feat_retrieval_depth` | Regex count matching on text | O(N) | Offline | Low | P0 |
| `feat_eval_rigor` | Regex count matching on text | O(N) | Offline | Low | P0 |
| `feat_product_exposure` | Text & metadata inference (size, keywords) | O(N) | Offline | Low | P1 |
| `feat_wrapper_ai` (Penalty) | Skill array parsing | O(N) | Offline | Low | P1 |
| `contradiction_score` | Temporal overlap + logical behavioral checks | O(N) | Offline | Low | P0 |
| `behavioral_mult` | Float math on normalized signals | O(1) per cand | Online | Low | P0 |

---

### Final Answer: Maximizing NDCG@10 and NDCG@50

**"If you were trying to maximize NDCG@10 and NDCG@50 for this specific competition, what exact feature set and weighting scheme would you deploy?"**

To maximize Top-10 and Top-50 metrics specifically, **I would deploy the Multiplicative Heuristic Formula above, entirely avoiding Learning-to-Rank (LTR).**

**Why?**
NDCG@10 is extremely sensitive. A single honeypot or false-positive candidate in your top 10 tanks your score. If you train an LTR model on synthetic heuristic labels, the model will inevitably "smooth out" your hard logic, occasionally letting a honeypot slip through or failing to heavily penalize a 90-day notice period. 

By using the **Multiplicative Heuristic Formula**, you guarantee absolute precision:
1.  **Honeypots are mathematically guaranteed to score near 0** (due to `exp(-contradiction_score)`).
2.  **Unreachable candidates are guaranteed to score near 0** (due to `Behavioral_Mult`).
3.  **The "Wrapper AI" penalty guarantees strict alignment with the JD's anti-persona.**

You extract all these features offline using Polars, save them as a Parquet file, and perform a FAISS inner-product search online to get the top 2000 semantic matches. Then, you simply apply the multiplication in Pandas/Numpy over those 2000 rows. This guarantees a highly precise Top 10, executes in under 2 seconds (safely under the 5-minute limit), and uses minimal RAM.
