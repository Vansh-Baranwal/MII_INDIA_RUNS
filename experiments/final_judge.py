import polars as pl
import numpy as np
import math
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

artifacts_dir = Path(r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS\artifacts")

print("--- Initializing Scenario D ---")
jd_text = "Senior AI Engineer Search Ranking Retrieval Embeddings NDCG Vector Databases"
model = SentenceTransformer('all-MiniLM-L6-v2')
jd_emb = model.encode([jd_text])
faiss.normalize_L2(jd_emb)

indices_set = set()
for prefix, k in [('recent', 2500), ('last_two', 1500), ('full', 1000)]:
    idx_path = artifacts_dir / f'candidates_{prefix}.faiss'
    index = faiss.read_index(str(idx_path))
    if index.ntotal < k: k = index.ntotal
    if k > 0:
        _, I = index.search(jd_emb, k)
        indices_set.update(I[0].tolist())

df = pl.read_parquet(artifacts_dir / 'features.parquet', glob=False).with_row_index("faiss_id")
df_pool = df.filter(pl.col("faiss_id").is_in(list(indices_set)))

emb_recent = np.load(artifacts_dir / 'embeddings_recent.npy')
emb_last_two = np.load(artifacts_dir / 'embeddings_last_two.npy')
emb_full = np.load(artifacts_dir / 'embeddings_full.npy')

faiss.normalize_L2(emb_recent)
faiss.normalize_L2(emb_last_two)
faiss.normalize_L2(emb_full)

pool_ids = df_pool['faiss_id'].to_list()
df_pool = df_pool.with_columns([
    pl.Series("sim_recent", (emb_recent[pool_ids] @ jd_emb[0]).tolist()),
    pl.Series("sim_last_two", (emb_last_two[pool_ids] @ jd_emb[0]).tolist()),
    pl.Series("sim_full", (emb_full[pool_ids] @ jd_emb[0]).tolist()),
])

df_pool = df_pool.with_columns(
    Base_Score = (0.55 * pl.col("sim_recent")) + (0.30 * pl.col("sim_last_two")) + (0.15 * pl.col("sim_full")),
    Technical_Multiplier = 1.0 + (
        0.25 * pl.col("feat_search_relevance_evidence") +
        0.20 * pl.col("feat_ranking_depth") + 
        0.15 * pl.col("feat_retrieval_depth") +
        0.15 * pl.col("feat_evaluation_rigor") +
        0.35 * pl.col("feat_builder_score")
    ),
    Trajectory_Multiplier = pl.col("feat_product_exposure") + pl.col("feat_trajectory_transition"),
    Raw_Behavioral_Multiplier = pl.col("feat_availability_score") + pl.col("feat_saved_boost") + pl.col("feat_search_appearance_boost"),
    Persona_Penalty = pl.col("feat_wrapper_ai_only") * pl.col("feat_architect_no_coding"),
    Honeypot_Decay = pl.col("contradiction_score").map_elements(lambda x: math.exp(-x), return_dtype=pl.Float64)
)

df_pool = df_pool.with_columns(
    Behavioral_Multiplier = 1.0 + (pl.col("Raw_Behavioral_Multiplier") - 1.0) * 0.25
)

df_pool = df_pool.with_columns(
    Final_Score = (
        pl.col("Base_Score") *
        pl.col("Technical_Multiplier") *
        pl.col("feat_verified_search_skill") *
        pl.col("Trajectory_Multiplier") *
        pl.col("Behavioral_Multiplier") *
        pl.col("Persona_Penalty") *
        pl.col("Honeypot_Decay")
    )
)

top30 = df_pool.sort(["Final_Score", "candidate_id"], descending=[True, False]).head(30)

print("\n--- Feature Decomposition Table (Top 30) ---")
print("ID | Title | YoE | Base | SearchRel | RetDepth | RankDepth | Eval | Builder | ProdExp | Final")
for i, row in enumerate(top30.iter_rows(named=True)):
    print(f"R{i+1}: {row['candidate_id']} | {row['current_title']} | {row['years_of_experience']} | "
          f"{row['Base_Score']:.3f} | {row['feat_search_relevance_evidence']:.3f} | {row['feat_retrieval_depth']:.3f} | "
          f"{row['feat_ranking_depth']:.3f} | {row['feat_evaluation_rigor']:.3f} | {row['feat_builder_score']:.3f} | "
          f"{row['feat_product_exposure']:.3f} | {row['Final_Score']:.4f}")

# Rules
rule1 = []
rule2 = []
rule3 = []

for i, row in enumerate(top30.iter_rows(named=True)):
    rank = i + 1
    t = row['current_title'].lower() if row['current_title'] else ""
    
    # Rule 1
    if row['feat_builder_score'] > 0.80 and row['feat_search_relevance_evidence'] < 0.20:
        rule1.append((rank, row))
        
    # Rule 2
    if 'data scientist' in t and row['feat_ranking_depth'] < 0.20 and row['feat_retrieval_depth'] < 0.20:
        rule2.append((rank, row))
        
    # Rule 3
    if row['feat_builder_score'] < 0.30 and rank <= 15:
        rule3.append((rank, row))

print("\n--- Flagged Candidates ---")
print("\nRule 1: Builder > 0.80 AND SearchRel < 0.20")
for r, row in rule1:
    print(f"Rank {r} ({row['candidate_id']} - {row['current_title']}): Builder={row['feat_builder_score']:.3f}, SearchRel={row['feat_search_relevance_evidence']:.3f}")
    print(f"  -> Elevated by: Base_Score={row['Base_Score']:.3f}, ProdExp={row['feat_product_exposure']:.3f}, TrajectoryMult={row['Trajectory_Multiplier']:.3f}")

print("\nRule 2: Data Scientist AND RankDepth < 0.20 AND RetDepth < 0.20")
for r, row in rule2:
    print(f"Rank {r} ({row['candidate_id']} - {row['current_title']}): RankDepth={row['feat_ranking_depth']:.3f}, RetDepth={row['feat_retrieval_depth']:.3f}")
    print(f"  -> Elevated by: Base_Score={row['Base_Score']:.3f}, Builder={row['feat_builder_score']:.3f}, ProdExp={row['feat_product_exposure']:.3f}")

print("\nRule 3: Builder < 0.30 AND Rank <= 15")
for r, row in rule3:
    print(f"Rank {r} ({row['candidate_id']} - {row['current_title']}): Builder={row['feat_builder_score']:.3f}")
    print(f"  -> Elevated by: Base_Score={row['Base_Score']:.3f}, SearchRel={row['feat_search_relevance_evidence']:.3f}, BehavMult={row['Behavioral_Multiplier']:.3f}")
