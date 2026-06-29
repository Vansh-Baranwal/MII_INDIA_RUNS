# Red Team Report: Breaking the Redrob Ranking System

This document acts as the evaluation team's adversarial review of the current ranking architecture. The goal is to shatter the proposed heuristics, expose critical blind spots, and guarantee that the final system survives the hidden dataset's edge cases.

---

## 1. Critical Failure Modes & Unfair Penalties

### A. The "Modest Staff Engineer" (Ranked Too Low)
*   **Profile:** 9 YOE. Title: "Software Engineer". Description: "Core contributor to distributed search backend in C++. Built custom inverted index and relevance pipelines."
*   **Why it fails:** They don't use buzzwords. They didn't mention "FAISS," "SentenceTransformers," "NDCG," or "Embeddings." The regex multipliers (`feat_retrieval_depth`, `feat_evaluation_rigor`) all return 0. The system assumes they are a generic backend SWE.
*   **Severity:** **High** (Destroys NDCG@10 by burying the true tier-5 veterans).
*   **Fix:** Expand the semantic net to include generic search concepts ("inverted index", "relevance pipeline", "Lucene", "Solr", "query understanding"). Use an LLM *offline* to tag profiles with a `true_seniority` score that doesn't rely on buzzwords.

### B. The "Academic DevRel / Educator" (Ranked Too High)
*   **Profile:** 4 YOE. Title: "Developer Advocate". Description: "I teach courses on embeddings, FAISS, learning-to-rank, NDCG, and semantic search. I build tutorials for retrieval systems."
*   **Why it fails:** They hit a 100% match on the MiniLM semantic baseline. They trigger every single regex multiplier. They have massive GitHub activity and high engagement. The system ranks them #1.
*   **Severity:** **Critical** (Guaranteed false positive in Top 10).
*   **Fix:** Add an explicit anti-persona penalty (`feat_educator`) searching for "DevRel", "Developer Advocate", "course", "tutorial", "teach". 

### C. The "Big Tech Non-Searcher" (Ranked Too High)
*   **Profile:** 6 YOE at Google. Title: "ML Engineer". Description: "Worked on YouTube ad placement using generic ML. Evaluated models with MAP."
*   **Why it fails:** High `product_exposure` (Google), triggers `feat_eval_rigor` (MAP), and "ML Engineer" sounds semantic. But they don't do *search/retrieval*.
*   **Severity:** **Medium**.
*   **Fix:** Require a strict chronological intersection. The retrieval keywords *must* co-occur with the most recent job, not just float anywhere in the profile.

### D. The "Walmart Labs" Phenomenon (Ranked Too Low)
*   **Profile:** 6 YOE ML Engineer at Walmart Global Tech building massive e-commerce search.
*   **Why it fails:** The `feat_product_exposure` assumes large `company_size` (10,000+) = IT Service Firm (TCS/Wipro) unless explicitly caught. Walmart is huge but has a massive product org. The candidate gets hit with the service firm penalty.
*   **Severity:** **High**.
*   **Fix:** The product/service heuristic must look at the `industry` flag ("Retail", "E-commerce") and not just rely on company size and a WITCH denylist.

### E. The "Time-Traveling Graduate" (Honeypot Bypass)
*   **Profile:** `yoe = 8`. Job history has 1 job lasting 96 months. Perfectly matches YOE.
*   **Why it fails:** The `contradiction_score` checks if durations exceed YOE. It passes. BUT, their `education.end_year` is 2024. They claim 8 years of experience but graduated 2 years ago.
*   **Severity:** **Critical** (Automatic disqualification if a honeypot slips into top 100).
*   **Fix:** Add `education_temporal_contradiction`. If `(current_year - education_end_year) < yoe - 2` (allowing 2 years of internships), tag as honeypot.

---

## 2. Top 20 Failure Modes Ranked by Risk

| Rank | Failure Mode | Vulnerability Type | Impact |
| :--- | :--- | :--- | :--- |
| **1** | **Time-Traveling Graduate** (YOE vs Edu Date) | Honeypot Bypass | Disqualification |
| **2** | **DevRel/Educator Keyword Stuffer** | False Positive | Kills NDCG@10 |
| **3** | **Modest C++ Search Veteran** | False Negative | Kills NDCG@10 |
| **4** | **Salary vs YOE Trap** (1 YOE, 80LPA) | Honeypot Bypass | Disqualification |
| **5** | **The "Ancient Searcher"** (Built search in 2014, generic backend since) | False Positive | Hurts NDCG@50 |
| **6** | **Large E-Commerce falsely flagged as IT Services** | False Negative | Hurts NDCG@50 |
| **7** | **Semantic Dilution** (MiniLM failing on long generic paragraphs) | Core Algorithm Bias | Kills MAP |
| **8** | **The "Data Scientist" Mismatch** (Heavy ML, zero engineering) | False Positive | Hurts NDCG@10 |
| **9** | **The Zero-Commit GitHub Score** (Score 100 but 0 actual repos) | Honeypot Bypass | Disqualification |
| **10** | **The Serial Intern** (6 jobs of 3 months = 1.5 YOE, flags as job hopper) | Unfair Penalty | Low Risk |
| **11** | **Spelling Errors in Keywords** ("elastic search" vs "elasticsearch") | Extraction Failure | Hurts NDCG@50 |
| **12** | **The "Manager" who still codes** (Penalized by title) | Unfair Penalty | Medium Risk |
| **13** | **Title Inflation** ("VP of AI" at a 1-person startup) | False Positive | Hurts NDCG@10 |
| **14** | **Academic with 1-month industry internship** (Bypasses research penalty) | False Positive | Medium Risk |
| **15** | **Missing Notice Period** (Null values break math) | Pipeline Crash | Format Rejection |
| **16** | **High Engagement, Zero Competence** (Multiplier inflates trash) | Formula Flaw | Kills NDCG@50 |
| **17** | **The "RAG" only candidate** (Lacks dense retrieval fundamentals) | False Positive | Hurts NDCG@10 |
| **18** | **Open Source Contributors** (No formal jobs, massive impact) | False Negative | Rare but High Impact |
| **19** | **Behavioral Contradiction** (100% Interview complete, 0 applications) | Honeypot Bypass | Disqualification |
| **20** | **The "Consulting to Product" penalty misfire** | Logic Error | Hurts MAP |

---

## 3. Improving Metrics

### A. Features that materially improve NDCG@10
To get the absolute best candidates at the very top:
1.  **`feat_recency_weighted_relevance`**: Don't just embed the whole profile. Embed *only* the most recent job description. If their search experience was 6 years ago, they shouldn't be rank 1.
2.  **`feat_hardcore_engineering`**: Look for low-level systems languages (C++, Rust, Go) combined with search. True ranking engineers often work below the Python layer.
3.  **`feat_leadership_velocity`**: Did they go from SWE -> Senior SWE -> Staff SWE at the same company? This is the strongest anti-job-hopper signal and highly correlated with Tier 5.

### B. Features that materially improve NDCG@50
To ensure the "middle of the pack" is solid and relevant:
1.  **`feat_domain_synonyms`**: Expand regex to include "Lucene," "Solr," "BM25," "tf-idf," "query expansion," "LTR."
2.  **`feat_industry_alignment`**: Give a slight boost to candidates currently in "HR Tech", "Marketplaces", or "E-Commerce", as these match the Redrob use-case perfectly.

### C. Features to REMOVE (Noise Generators)
1.  **`feat_vectordb_exposure`**: Drop it entirely. It heavily biases towards junior developers who list every tool they've ever touched.
2.  **`feat_prod_ml_scale` (as currently defined)**: "Deployment" and "Inference" are too generic. Almost every modern SWE claims they deploy things.

---

## 4. Final Pre-Submission Checklist

Before running `validate_submission.py` and uploading, the pipeline must pass these checks:

- [ ] **Honeypot Education Check:** Does `education.end_year` logically align with `years_of_experience`?
- [ ] **Honeypot Salary Check:** Is `expected_salary_range_inr_lpa.min` > 40 while `years_of_experience` < 3? (Instant reject).
- [ ] **Honeypot Engagement Check:** Does `profile_views_received_30d` == 0 while `recruiter_response_rate` > 0? (Impossible state).
- [ ] **DevRel/Educator Filter:** Are terms like "Evangelist", "DevRel", "Course", "Tutor" heavily penalized?
- [ ] **Recency Bias Check:** Is the semantic similarity driven by their current job, or a job from 2018?
- [ ] **Missing Value Safeguards:** Are `null` notice periods defaulting to 30 days rather than breaking the math or defaulting to 0 (which would give them a massive boost)?
- [ ] **Reasoning Sanity Check:** Did the reasoning generator accidentally output "Strong product company exposure" for someone whose only job is TCS? (Triggers Stage 4 rejection).
- [ ] **Rank Distribution Check:** Plot the scores. Are they clustered between 0.99 and 1.00? If so, the multipliers are broken. Spread should be wide.
- [ ] **Tie-Breaker Determinism:** If two candidates have the exact same final score, does the script break the tie deterministically (e.g., sort by `candidate_id`)? `rank` must be unique 1-100.
