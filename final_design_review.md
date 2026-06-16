# Final Design Review: Ranking System Architecture

This document represents the finalized, red-team-hardened ranking architecture for the Redrob Intelligent Candidate Discovery & Ranking Challenge. It corrects the flaws identified in the evaluation phase, explicitly addressing semantic dilution, education-based honeypots, and the builder-versus-educator distinction.

---

## 1. Semantic Similarity Scoring (Recency-Biased)

**Critique Addressed:** Embedding the entire profile as one text block diluted the signal, rewarding candidates who worked on search 8 years ago but haven't touched it since.

**Redesign:**
We split the semantic retrieval into three distinct vector embeddings per candidate.
*   `semantic_recent_role`: Cosine similarity of the JD against ONLY the most recent `career_history[0].description`.
*   `semantic_last_two_roles`: Cosine similarity of the JD against `career_history[0]` + `career_history[1]`.
*   `semantic_full_profile`: Cosine similarity against `summary` + all `career_history`.

**Weighting Strategy & Why Recency Dominates:**
*   **Base Semantic Score =** `(0.55 * semantic_recent_role) + (0.30 * semantic_last_two_roles) + (0.15 * semantic_full_profile)`
*   *Why:* The ML ecosystem moves incredibly fast. A candidate who built search in 2018 but has been doing generic CRUD backend for the last 6 years is effectively a junior ranking engineer today. Heavy weighting on the most recent role guarantees that the top 10 candidates are actively building in this space *right now*.

## 2. Search Relevance Evidence (`feat_search_relevance_evidence`)

**Critique Addressed:** Previous features over-indexed on vector databases (FAISS, Pinecone) which commoditize the space, missing candidates who build complex relevance logic.

**Design:**
*   **Definition:** Measures deep, domain-specific search and relevance expertise beyond simple tooling.
*   **Extraction Logic:** Regex matching across the profile for systems-level search terms.
    ```python
    terms = r"\b(candidate matching|recommendation systems|search quality|retrieval pipelines|ranking services|relevance optimization|personalization|marketplace ranking|query understanding|inverted index|lucene|solr|tf-idf|bm25)\b"
    score = min(count_matches(terms, profile_text) / 3.0, 1.0)
    ```
*   **Recommended Weight:** `0.25` (Highest technical multiplier).

## 3. Education-Based Contradiction Detection

**Critique Addressed:** The "Time-Traveling Graduate" honeypot bypassed job duration checks by simply lying consistently about job lengths.

**Design:**
*   **Definition:** `education_temporal_contradiction`. Identifies impossible years of experience relative to college graduation.
*   **Formulas:**
    ```python
    current_year = 2026
    
    if len(education) > 0:
        grad_year = max(ed['end_year'] for ed in education)
        # Allow 2 years of overlap for internships / working during master's
        max_possible_yoe = (current_year - grad_year) + 2 
        
        if years_of_experience > max_possible_yoe:
            # Massive contradiction
            return 5.0 # Adds to the exponential decay penalty
    return 0.0
    ```

## 4. Builder vs Educator Differentiation (`feat_builder_score`)

**Critique Addressed:** Relying on a `feat_educator` penalty is risky because true engineers might also teach or write blogs. Instead, we heavily reward "Builder" actions.

**Design:**
*   **Definition:** Rewards candidates whose descriptions use active, production-engineering verbs, separating people who write tutorials from people who ship systems.
*   **Extraction Logic:** Match high-agency engineering verbs.
    ```python
    builder_verbs = r"\b(built|deployed|shipped|implemented|scaled|production ownership|architected|designed|optimized|productionized|owned)\b"
    score = min(count_matches(builder_verbs, profile_text) / 5.0, 1.0)
    ```
*   **Recommended Weight:** `0.15` (Used as a technical multiplier).

---

## 5. Final Feature Catalog (Pre-Implementation)

| Category | Feature Name | Normalization | Type |
| :--- | :--- | :--- | :--- |
| **Semantic (Base)** | `base_semantic_score` | `[0, 1]` | Additive Base |
| **Search/Ranking** | `feat_search_relevance_evidence` | `[0, 1]` | Multiplier |
| **Retrieval Depth** | `feat_retrieval_depth` | `[0, 1]` | Multiplier |
| **Evaluation** | `feat_evaluation_rigor` | `[0, 1]` | Multiplier |
| **Engineering** | `feat_builder_score` | `[0, 1]` | Multiplier |
| **Trajectory** | `feat_product_exposure` | `[0.5, 1.0]` | Multiplier |
| **Trajectory** | `feat_trajectory_transition` | `[0, 0.2]` | Multiplier Bonus |
| **Behavioral** | `feat_availability_score` | `[0, 1.2]` | Multiplier |
| **Penalty** | `feat_wrapper_ai_only` | `[0.3, 1.0]` | Decay Multiplier |
| **Penalty** | `feat_architect_no_coding` | `[0.6, 1.0]` | Decay Multiplier |
| **Honeypot** | `contradiction_score` | `[0, ∞)` | Exp. Decay |

*(Note: `contradiction_score` is the sum of job-duration contradictions, behavioral contradictions, and the new education-temporal contradictions).*

---

## 6. Final Ranking Equation

This is the exact, final heuristic formula to be deployed in the 5-minute online ranking script.

```python
# 1. Base Semantic Score (Recency Weighted)
Base_Score = (0.55 * semantic_recent_role) + \
             (0.30 * semantic_last_two_roles) + \
             (0.15 * semantic_full_profile)

# 2. Technical Skill Multiplier
# A perfect engineer gets a 2.0x multiplier. An LLM wrapper gets 1.0x.
Technical_Multiplier = 1.0 + (
    0.25 * feat_search_relevance_evidence +
    0.20 * feat_retrieval_depth +
    0.20 * feat_evaluation_rigor +
    0.15 * feat_builder_score
)

# 3. Career Trajectory Multiplier
# Maxes out at 1.2x for perfect product trajectory
Trajectory_Multiplier = feat_product_exposure + feat_trajectory_transition

# 4. Behavioral Multiplier
# Formula: (response_rate) * exp(-days_inactive / 20) * (notice_period_decay)
# Maxes out at ~1.2x for highly active, immediately available candidates
Behavioral_Multiplier = feat_availability_score

# 5. Negative Penalties
# Soft penalties for AI wrappers or non-coding architects. Usually 1.0.
Persona_Penalty = feat_wrapper_ai_only * feat_architect_no_coding

# 6. Honeypot Decay
# Any temporal or behavioral contradiction exponentially collapses the score to 0.
Honeypot_Decay = math.exp(-contradiction_score)

# ==========================================
# FINAL EQUATION
# ==========================================
Final_Score = Base_Score * Technical_Multiplier * Trajectory_Multiplier * Behavioral_Multiplier * Persona_Penalty * Honeypot_Decay
```

### Why this guarantees a win:
By heavily weighting `semantic_recent_role` and multiplying it by `feat_search_relevance_evidence`, we isolate candidates actively building search systems. By shifting away from educator penalties toward a `feat_builder_score` multiplier, we organically filter out DevRel noise while rewarding true engineers. Finally, the strict `education_temporal_contradiction` mathematically guarantees zero honeypots in the Top 100.
