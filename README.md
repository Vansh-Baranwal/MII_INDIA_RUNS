# India Runs: Search Quality Ranking Pipeline

This repository contains the end-to-end ranking pipeline for identifying Principal Search Quality Engineers for Redrob. It leverages a combination of structured feature engineering, semantic density checks, and FAISS-based vector search to mathematically rank candidates based on their technical depth, algorithmic purity, and search trajectory.

## Problem Statement

Identifying Principal-level Search Engineers requires more than generic keyword matching. An elite Search Quality engineer must possess highly specialized knowledge across four vectors:
1. **Retrieval Infrastructure** (Vector databases, approximate nearest neighbors)
2. **Ranking Architectures** (Learning to Rank, LambdaMART)
3. **Relevance Evaluation** (NDCG, MRR, Interleaving)
4. **Production Engineering** (Architecting and scaling systems)

This pipeline converts raw candidate JSON blobs into a highly calibrated ordinal ranking that specifically targets these four sub-domains, penalizing generic software engineering backgrounds while aggressively rewarding explicit search domain expertise.

## Architecture Diagram

```
Raw Candidates -> [parse_candidates.py] -> parsed_candidates.parquet
                       |
                       v
              [feature_engineering.py] -> features.parquet
                       |
                       v
                 [embeddings.py] -> embeddings_*.npy
                       |
                       v
                [build_faiss.py] -> candidates_*.faiss
                       |
                       v
                    [rank.py] -> submission.csv & top20.csv
```

## Core Pipeline Components

### 1. Feature Engineering (`feature_engineering.py`)
Extracts mathematical scores for `Builder`, `Retrieval`, `Ranking`, and `Evaluation` by applying specialized term-frequency dictionaries over the raw profile texts. It calculates a `Contradiction_Score` to penalize candidates whose expected salaries or graduation years contradict their claimed years of experience.

### 2. Semantic Embedding (`embeddings.py`)
Converts the candidate's career histories (split by recent roles, last two roles, and full history) into dense vectors using the `all-MiniLM-L6-v2` SentenceTransformer.

### 3. Retrieval Pipeline (`build_faiss.py`)
Constructs scalable `IndexFlatIP` FAISS indices for the generated embeddings, allowing ultra-fast similarity matching against an ideal Job Description (JD) vector.

### 4. Ranking Pipeline (`rank.py`)
Executes the final mathematical scoring:
*   **Base Score:** A weighted combination of FAISS similarity scores (`0.55 * Recent + 0.30 * Last Two + 0.15 * Full`).
*   **Technical Multiplier:** Rewards explicit Search, Ranking, Retrieval, and Evaluation skills.
*   **Trajectory Multiplier:** Rewards engineers who show sustained growth as "builders."
*   **Elite Exemption:** Completely removes contradiction penalties for the top 1% of mathematically proven Search builders.
*   **Final Formula:** `Final_Score = Core_Mult * (1.0 + 0.20 * Base_Score)` (Variant D Architecture).

## Installation

```bash
pip install -r requirements.txt
```

## Execution Instructions

To run the entire pipeline end-to-end, execute the provided PowerShell script:

```bash
./run_pipeline.ps1
```

*(Note: The first run will download the `sentence-transformers` model from Hugging Face if not cached).*

## Output Description

*   `submission.csv`: The final Top 100 candidates formatted for submission, containing the `candidate_id`, `rank`, `score`, and dynamically generated judge `reasoning`.
*   `artifacts/`: Temporary cache for embeddings, FAISS indices, and parquet tables.

## Repository Structure

```
├── .gitignore
├── README.md
├── requirements.txt
├── run_pipeline.ps1
├── docs/
│   └── (Architectural audits, design reviews, code reviews)
├── experiments/
│   └── (Ablation studies, evaluation tests, experimental scripts)
└── src/
    ├── offline/
    │   ├── parse_candidates.py
    │   ├── feature_engineering.py
    │   ├── embeddings.py
    │   └── build_faiss.py
    └── online/
        ├── rank.py
        └── reasoning.py
```
