import polars as pl
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import json

artifacts_dir = Path(r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS\artifacts")

# --- Reconstruct ranked pool ---
jd_text = "Senior AI Engineer Search Ranking Retrieval Embeddings NDCG Vector Databases"
model = SentenceTransformer('all-MiniLM-L6-v2')
jd_emb = model.encode([jd_text])
faiss.normalize_L2(jd_emb)

indices_set = set()
for prefix, k in [('recent', 2500), ('last_two', 1500), ('full', 1000)]:
    idx_path = artifacts_dir / f'candidates_{prefix}.faiss'
    index = faiss.read_index(str(idx_path))
    actual_k = min(k, index.ntotal)
    if actual_k > 0:
        _, I = index.search(jd_emb, actual_k)
        indices_set.update(I[0].tolist())

df_feat = pl.read_parquet(artifacts_dir / 'features.parquet', glob=False).with_row_index("faiss_id")
df_pool = df_feat.filter(pl.col("faiss_id").is_in(list(indices_set)))

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

df_parsed = pl.read_parquet(artifacts_dir / 'parsed_candidates.parquet', glob=False).select(["candidate_id", "current_title", "years_of_experience"])
df_pool = df_pool.join(df_parsed, on="candidate_id", how="left")
# Use parsed YoE if feat YoE is not available or rename it
# Actually features.parquet already has years_of_experience, but let's make sure
if 'years_of_experience_right' in df_pool.columns:
    df_pool = df_pool.drop('years_of_experience_right')

# Common fields
df_pool = df_pool.with_columns([
    pl.col("sim_recent").alias("sim_recent"),
    (0.55*pl.col("sim_recent") + 0.30*pl.col("sim_last_two") + 0.15*pl.col("sim_full")).alias("Base_Score"),
    (1.0 + (pl.col("feat_availability_score") + pl.col("feat_saved_boost") + pl.col("feat_search_appearance_boost") - 1.0)*0.25).alias("Behavioral_Multiplier"),
    (pl.col("feat_wrapper_ai_only") * pl.col("feat_architect_no_coding")).alias("Persona_Penalty"),
    ((-pl.col("contradiction_score")).exp()).alias("Honeypot_Decay"),
    (1.0 + (0.25*pl.col("feat_search_relevance_evidence") + 0.20*pl.col("feat_ranking_depth") + 0.15*pl.col("feat_retrieval_depth") + 0.15*pl.col("feat_evaluation_rigor") + 0.35*pl.col("feat_builder_score"))).alias("Technical_Multiplier")
])

# Helper for parsing titles
def get_title_stats(titles):
    search = sum(1 for t in titles if any(k in t.lower() for k in ['search', 'ranking', 'relevance', 'ir engineer', 'information retrieval']))
    recsys = sum(1 for t in titles if any(k in t.lower() for k in ['recommendation', 'recsys']))
    nlp = sum(1 for t in titles if any(k in t.lower() for k in ['nlp', 'natural language']))
    cv = sum(1 for t in titles if any(k in t.lower() for k in ['computer vision', 'vision']))
    junior = sum(1 for t in titles if 'junior' in t.lower() or 'associate' in t.lower())
    research = sum(1 for t in titles if 'research' in t.lower() or 'scientist' in t.lower() and 'data' not in t.lower())
    return search, recsys, nlp, junior, research, cv

target_candidates = ["CAND_0092278", "CAND_0077337", "CAND_0060072", "CAND_0083307", "CAND_0018499"]

def simulate_scenario(name, traj_expr):
    df_sim = df_pool.with_columns(
        traj_expr.alias("Trajectory_Multiplier")
    )
    df_sim = df_sim.with_columns(
        (pl.col("Base_Score") * pl.col("Technical_Multiplier") * pl.col("feat_verified_search_skill") * pl.col("Trajectory_Multiplier") * pl.col("Behavioral_Multiplier") * pl.col("Persona_Penalty") * pl.col("Honeypot_Decay")).alias("Final_Score")
    )
    df_ranked = df_sim.sort(["Final_Score","candidate_id"], descending=[True,False]).with_row_index("rank")
    
    top20 = df_ranked.head(20)
    
    b_mean = top20['feat_builder_score'].mean()
    r_mean = top20['feat_retrieval_depth'].mean()
    s_mean = top20['feat_search_relevance_evidence'].mean()
    yoe_mean = top20['years_of_experience'].mean()
    
    titles = top20['current_title'].fill_null("").to_list()
    s_c, r_c, n_c, j_c, res_c, cv_c = get_title_stats(titles)
    
    print(f"\n{'='*100}")
    print(f"  {name}")
    print(f"{'='*100}")
    print(f"  Avg Builder Score:    {b_mean:.3f}")
    print(f"  Avg Retrieval Depth:  {r_mean:.3f}")
    print(f"  Avg Search Relevance: {s_mean:.3f}")
    print(f"  Avg Years Experience: {yoe_mean:.1f}")
    print(f"  Titles:")
    print(f"    Search/Ranking: {s_c}")
    print(f"    Recommendation: {r_c}")
    print(f"    NLP:            {n_c}")
    print(f"    Junior:         {j_c}")
    print(f"    Research:       {res_c}")
    print(f"    Computer Vis:   {cv_c}")
    
    print(f"\n  Target Candidates:")
    for cid in target_candidates:
        row = df_ranked.filter(pl.col("candidate_id") == cid)
        if len(row) > 0:
            rank = row['rank'][0] + 1
            print(f"    {cid:<15} Rank: {rank:>4}")
        else:
            print(f"    {cid:<15} Rank:  N/A")
            
    print(f"\n  Top 5 Candidates:")
    for row in top20.head(5).iter_rows(named=True):
        print(f"    {row['rank']+1:>2} | {row['candidate_id']} | {row['current_title']:<35} | B={row['feat_builder_score']:.2f}")
    
    return b_mean, r_mean

# --- Scenarios ---
# A: Current
expr_A = pl.col("feat_product_exposure") + pl.col("feat_trajectory_transition")
# B: trajectory_transition = 0
expr_B = pl.col("feat_product_exposure")
# C: product_exposure = 0.50
expr_C = pl.lit(0.50) + pl.col("feat_trajectory_transition")
# D: Remove entirely (i.e. Trajectory_Multiplier = 1.0)
expr_D = pl.lit(1.0)
# E: 0.5 + 0.5 * builder_score
expr_E = 0.50 + 0.50 * pl.col("feat_builder_score")

results = []
results.append(("Scenario A (Current)", simulate_scenario("Scenario A (Current)", expr_A)))
results.append(("Scenario B (TrajTrans=0)", simulate_scenario("Scenario B (TrajTrans=0)", expr_B)))
results.append(("Scenario C (ProdExp=0.5)", simulate_scenario("Scenario C (ProdExp=0.5)", expr_C)))
results.append(("Scenario D (Remove Traj Mult)", simulate_scenario("Scenario D (Remove Traj Mult)", expr_D)))
results.append(("Scenario E (0.5 + 0.5*Builder)", simulate_scenario("Scenario E (0.5 + 0.5*Builder)", expr_E)))

print("\nDone.")
