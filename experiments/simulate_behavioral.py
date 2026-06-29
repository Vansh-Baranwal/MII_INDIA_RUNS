import polars as pl
import numpy as np
import math
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

artifacts_dir = Path(r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS\artifacts")

# Get Baseline Top 20 to compare
df_debug = pl.read_parquet(artifacts_dir / 'debug_top200.parquet', glob=False)
baseline_top20_ids = df_debug.head(20)['candidate_id'].to_list()

print("--- Recomputing Union Pool for Simulations ---")
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

# Fixed builder_weight = 0.35 from previous step
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

def run_scenario(name, behavioral_expr):
    df_sim = df_pool.with_columns(
        Behavioral_Multiplier = behavioral_expr
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
    
    top20 = df_sim.sort(["Final_Score", "candidate_id"], descending=[True, False]).head(20)
    top20_ids = top20['candidate_id'].to_list()
    
    entering = set(top20_ids) - set(baseline_top20_ids)
    leaving = set(baseline_top20_ids) - set(top20_ids)
    
    avg_builder = top20['feat_builder_score'].mean()
    avg_yoe = top20['years_of_experience'].mean()
    
    titles = top20['current_title'].fill_null("").str.to_lowercase().to_list()
    c_tech = sum(1 for t in titles if 'search' in t or 'nlp' in t or 'recommendation' in t)
    c_jun = sum(1 for t in titles if 'junior' in t or 'associate' in t)
    c_res = sum(1 for t in titles if 'research' in t)
    
    print(f"\n==========================================")
    print(f"Scenario {name}")
    print(f"==========================================")
    print("--- New Top 20 ---")
    for i, row in enumerate(top20.iter_rows(named=True)):
        print(f"Rank {i+1} | {row['candidate_id']} | Title: {row['current_title']} | Score: {row['Final_Score']:.4f} | Behav: {row['Behavioral_Multiplier']:.3f} | Builder: {row['feat_builder_score']:.3f}")
    
    print("\n--- Entering Top 20 ---")
    df_entering = top20.filter(pl.col('candidate_id').is_in(list(entering)))
    for row in df_entering.iter_rows(named=True):
        print(f"+ {row['candidate_id']} | {row['current_title']} | Builder: {row['feat_builder_score']:.3f}")
    
    print("\n--- Leaving Top 20 ---")
    df_leaving = df_debug.filter(pl.col('candidate_id').is_in(list(leaving)))
    for row in df_leaving.iter_rows(named=True):
        print(f"- {row['candidate_id']} | {row['current_title']} | Builder: {row['feat_builder_score']:.3f}")
        
    print("\n--- Metrics ---")
    print(f"Avg Builder Score: {avg_builder:.3f}")
    print(f"Avg YoE: {avg_yoe:.2f}")
    print(f"Search/NLP/Rec Titles: {c_tech}")
    print(f"Junior Titles: {c_jun}")
    print(f"Research Titles: {c_res}")


# A: capped at 1.05
# pl.col("Raw_Behavioral_Multiplier").clip_max(1.05) - Wait, polars has clip(lower_bound, upper_bound)
run_scenario("A (Cap 1.05)", pl.col("Raw_Behavioral_Multiplier").clip(lower_bound=0.0, upper_bound=1.05))

# B: capped at 1.00
run_scenario("B (Cap 1.00)", pl.col("Raw_Behavioral_Multiplier").clip(lower_bound=0.0, upper_bound=1.00))

# C: contribution * 0.50
# If Raw_Behavioral_Multiplier > 1.0, contribution is (Raw - 1.0). 
# If Raw_Behavioral_Multiplier < 1.0, wait, usually it's > 1.0. Let's just scale the distance from 1.0.
expr_c = 1.0 + (pl.col("Raw_Behavioral_Multiplier") - 1.0) * 0.50
run_scenario("C (Contrib 0.50)", expr_c)

# D: contribution * 0.25
expr_d = 1.0 + (pl.col("Raw_Behavioral_Multiplier") - 1.0) * 0.25
run_scenario("D (Contrib 0.25)", expr_d)

