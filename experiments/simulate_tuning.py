import polars as pl
import numpy as np
import math
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

artifacts_dir = Path(r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS\artifacts")

df_debug = pl.read_parquet(artifacts_dir / 'debug_top200.parquet', glob=False)
df_top50 = df_debug.head(50)

# 1. Count title seniority levels
titles = df_top50['current_title'].fill_null("").str.to_lowercase().to_list()
counts = {
    'Junior': sum(1 for t in titles if 'junior' in t or 'associate' in t),
    'Engineer': sum(1 for t in titles if 'engineer' in t and not any(x in t for x in ['senior', 'staff', 'principal', 'lead', 'junior', 'manager'])),
    'Senior Engineer': sum(1 for t in titles if 'senior' in t),
    'Staff': sum(1 for t in titles if 'staff' in t),
    'Principal': sum(1 for t in titles if 'principal' in t),
    'Lead': sum(1 for t in titles if 'lead' in t),
    'Manager': sum(1 for t in titles if 'manager' in t),
    'Director': sum(1 for t in titles if 'director' in t),
    'VP': sum(1 for t in titles if 'vp' in t or 'vice president' in t)
}

print("--- Title Seniority Levels (Top 50) ---")
for k, v in counts.items(): print(f"{k}: {v}")

print("\n--- Top 50 Candidate Metrics ---")
for row in df_top50.iter_rows(named=True):
    print(f"[{row['candidate_id']}] {row['current_title']} | YoE: {row['years_of_experience']} | Score: {row['Final_Score']:.4f} | Builder: {row['feat_builder_score']:.3f} | SR_Evidence: {row['feat_search_relevance_evidence']:.3f} | Depth: {row['feat_ranking_depth']:.3f}")

print("\n--- Identification Flags ---")
for i, row in enumerate(df_top50.iter_rows(named=True)):
    rank = i + 1
    t = row['current_title'].lower() if row['current_title'] else ""
    if ('junior' in t or 'associate' in t) and rank <= 20:
        print(f"Junior ranking high: {row['candidate_id']} (Rank {rank})")
    if 'research' in t and rank <= 20:
        print(f"Research-heavy ranking high: {row['candidate_id']} (Rank {rank})")
    if rank <= 20 and row['feat_builder_score'] < 0.25:
        print(f"Builder < 0.25 in Top 20: {row['candidate_id']} (Rank {rank}) - Builder: {row['feat_builder_score']:.3f}")
    if rank <= 20 and row['years_of_experience'] is not None and row['years_of_experience'] < 3:
        print(f"YoE < 3 in Top 20: {row['candidate_id']} (Rank {rank}) - YoE: {row['years_of_experience']}")

print("\n--- Recomputing Union Pool for Simulations ---")
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
    Trajectory_Multiplier = pl.col("feat_product_exposure") + pl.col("feat_trajectory_transition"),
    Behavioral_Multiplier = pl.col("feat_availability_score") + pl.col("feat_saved_boost") + pl.col("feat_search_appearance_boost"),
    Persona_Penalty = pl.col("feat_wrapper_ai_only") * pl.col("feat_architect_no_coding"),
    Honeypot_Decay = pl.col("contradiction_score").map_elements(lambda x: math.exp(-x), return_dtype=pl.Float64)
)

def get_top_20(builder_weight):
    # Adjust technical multiplier with new builder weight
    # Current rank.py had 0.15 for builder, but the weights in rank.py were:
    # 0.25 (search) + 0.20 (rank depth) + 0.15 (ret depth) + 0.15 (eval) + 0.15 (builder) = 0.90 
    # (Actually 0.9 + 1.0 = 1.9 max)
    
    # We will compute Technical_Multiplier by substituting the builder weight
    df_sim = df_pool.with_columns(
        Technical_Multiplier = 1.0 + (
            0.25 * pl.col("feat_search_relevance_evidence") +
            0.20 * pl.col("feat_ranking_depth") + 
            0.15 * pl.col("feat_retrieval_depth") +
            0.15 * pl.col("feat_evaluation_rigor") +
            builder_weight * pl.col("feat_builder_score")
        )
    )
    
    df_sim = df_sim.with_columns(
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
    
    return df_sim.sort(["Final_Score", "candidate_id"], descending=[True, False]).head(20)

print("\n--- Simulation: builder_weight = 0.30 ---")
top20_030 = get_top_20(0.30)
for i, row in enumerate(top20_030.iter_rows(named=True)):
    print(f"Rank {i+1} | {row['candidate_id']} | Title: {row['current_title']} | Builder: {row['feat_builder_score']:.3f} | Score: {row['Final_Score']:.4f}")

print("\n--- Simulation: builder_weight = 0.35 ---")
top20_035 = get_top_20(0.35)
for i, row in enumerate(top20_035.iter_rows(named=True)):
    print(f"Rank {i+1} | {row['candidate_id']} | Title: {row['current_title']} | Builder: {row['feat_builder_score']:.3f} | Score: {row['Final_Score']:.4f}")
