import polars as pl
import numpy as np
import math
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

artifacts_dir = Path(r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS\artifacts")

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
    Raw_Behavioral_Multiplier = pl.col("feat_availability_score") + pl.col("feat_saved_boost") + pl.col("feat_search_appearance_boost"),
    Persona_Penalty = pl.col("feat_wrapper_ai_only") * pl.col("feat_architect_no_coding"),
    Honeypot_Decay = pl.col("contradiction_score").map_elements(lambda x: math.exp(-x), return_dtype=pl.Float64)
)

df_pool = df_pool.with_columns(
    Behavioral_Multiplier = 1.0 + (pl.col("Raw_Behavioral_Multiplier") - 1.0) * 0.25
)

track_candidates = ['CAND_0032515', 'CAND_0086022', 'CAND_0043860', 'CAND_0043381', 'CAND_0069638', 'CAND_0030953', 'CAND_0036437']

def run_simulation(name, search_rel_weight):
    df_sim = df_pool.with_columns(
        Technical_Multiplier = 1.0 + (
            search_rel_weight * pl.col("feat_search_relevance_evidence") +
            0.20 * pl.col("feat_ranking_depth") + 
            0.15 * pl.col("feat_retrieval_depth") +
            0.15 * pl.col("feat_evaluation_rigor") +
            0.35 * pl.col("feat_builder_score")
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
    
    top20 = df_sim.sort(["Final_Score", "candidate_id"], descending=[True, False]).head(20)
    top100 = df_sim.sort(["Final_Score", "candidate_id"], descending=[True, False]).head(100)
    
    avg_builder = top20['feat_builder_score'].mean()
    avg_yoe = top20['years_of_experience'].mean()
    
    titles = top20['current_title'].fill_null("").str.to_lowercase().to_list()
    c_tech = sum(1 for t in titles if 'search' in t or 'nlp' in t or 'recommendation' in t)
    c_jun = sum(1 for t in titles if 'junior' in t or 'associate' in t)
    c_res = sum(1 for t in titles if 'research' in t)
    
    print(f"\n==========================================")
    print(f"Scenario: {name} (Weight: {search_rel_weight:.4f})")
    print(f"==========================================")
    print("--- Top 20 ---")
    for i, row in enumerate(top20.iter_rows(named=True)):
        print(f"Rank {i+1} | {row['candidate_id']} | {row['current_title']} | Score: {row['Final_Score']:.4f} | SR: {row['feat_search_relevance_evidence']:.3f} | Builder: {row['feat_builder_score']:.3f}")
    
    print("\n--- Tracked Candidates ---")
    for cand in track_candidates:
        cand_row = top100.filter(pl.col('candidate_id') == cand)
        if len(cand_row) > 0:
            rank = cand_row.row(0, named=True)
            # Find rank index
            rank_idx = -1
            for idx, r in enumerate(top100.iter_rows(named=True)):
                if r['candidate_id'] == cand:
                    rank_idx = idx + 1
                    break
            print(f"{cand} ({rank['current_title']}): Rank {rank_idx}")
        else:
            print(f"{cand}: Dropped out of Top 100")
            
    print("\n--- Metrics ---")
    print(f"Avg Builder Score: {avg_builder:.3f}")
    print(f"Avg YoE: {avg_yoe:.2f}")
    print(f"Search/NLP/Rec Titles: {c_tech}")
    print(f"Junior Titles: {c_jun}")
    print(f"Research Titles: {c_res}")

run_simulation("Current (+0%)", 0.25)
run_simulation("+25%", 0.25 * 1.25)
run_simulation("+50%", 0.25 * 1.50)
run_simulation("+75%", 0.25 * 1.75)
run_simulation("+100%", 0.25 * 2.00)

