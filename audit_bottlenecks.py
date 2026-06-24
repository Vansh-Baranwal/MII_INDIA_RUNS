import polars as pl
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

artifacts_dir = Path(r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS\artifacts")

# --- Reconstruct base pool ---
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

df_parsed = pl.read_parquet(artifacts_dir / 'parsed_candidates.parquet', glob=False).select(["candidate_id", "current_title"])
df_pool = df_pool.join(df_parsed, on="candidate_id", how="left")

# 1. Component calculations
df_pool = df_pool.with_columns([
    (0.55*pl.col("sim_recent") + 0.30*pl.col("sim_last_two") + 0.15*pl.col("sim_full")).alias("Base_Score"),
    ((0.4 * pl.col("feat_builder_score") + 0.3 * pl.col("feat_ranking_depth") + 0.3 * pl.col("feat_retrieval_depth")).clip(0.0, 1.0)).alias("feat_search_builder"),
    (1.0 + (pl.col("feat_availability_score") + pl.col("feat_saved_boost") + pl.col("feat_search_appearance_boost") - 1.0)*0.25).alias("Behavioral_Multiplier"),
    (pl.col("feat_wrapper_ai_only") * pl.col("feat_architect_no_coding")).alias("Persona_Penalty"),
    ((-0.10 * pl.col("contradiction_score")).exp()).alias("Honeypot_Decay")
])

df_pool = df_pool.with_columns([
    (0.50 + 0.50 * pl.col("feat_search_builder")).alias("Trajectory_Multiplier"),
    (1.0 + (0.25*pl.col("feat_search_relevance_evidence") + 0.20*pl.col("feat_ranking_depth") + 0.15*pl.col("feat_retrieval_depth") + 0.15*pl.col("feat_evaluation_rigor") + 0.35*pl.col("feat_builder_score"))).alias("Technical_Multiplier")
])

df_pool = df_pool.with_columns(
    (pl.col("Base_Score") * pl.col("Technical_Multiplier") * pl.col("feat_verified_search_skill") * pl.col("Trajectory_Multiplier") * pl.col("Behavioral_Multiplier") * pl.col("Persona_Penalty") * pl.col("Honeypot_Decay")).alias("Final_Score")
)

df_ranked = df_pool.sort(["Final_Score","candidate_id"], descending=[True,False]).with_row_index("rank")

top20 = df_ranked.head(20)

# Top 20 averages
t20_base = top20['Base_Score'].mean()
t20_builder = top20['feat_builder_score'].mean()
t20_sb = top20['feat_search_builder'].mean()
t20_ret = top20['feat_retrieval_depth'].mean()
t20_rank = top20['feat_ranking_depth'].mean()
t20_eval = top20['feat_evaluation_rigor'].mean()
t20_beh = top20['Behavioral_Multiplier'].mean()
t20_cont = top20['contradiction_score'].mean()

target_candidates = ["CAND_0083307", "CAND_0092278", "CAND_0018499", "CAND_0077337", "CAND_0019480"]

def get_simulated_rank(cid, mod_dict):
    df_sim = df_ranked.clone()
    # Apply mod_dict to the specific candidate
    for col, val in mod_dict.items():
        df_sim = df_sim.with_columns(
            pl.when(pl.col("candidate_id") == cid).then(val).otherwise(pl.col(col)).alias(col)
        )
    
    # Recalculate composites if needed
    if 'feat_search_builder' in mod_dict or 'feat_builder_score' in mod_dict or 'feat_ranking_depth' in mod_dict or 'feat_retrieval_depth' in mod_dict:
        df_sim = df_sim.with_columns(
            ((0.4 * pl.col("feat_builder_score") + 0.3 * pl.col("feat_ranking_depth") + 0.3 * pl.col("feat_retrieval_depth")).clip(0.0, 1.0)).alias("feat_search_builder")
        )
        df_sim = df_sim.with_columns(
            (0.50 + 0.50 * pl.col("feat_search_builder")).alias("Trajectory_Multiplier"),
            (1.0 + (0.25*pl.col("feat_search_relevance_evidence") + 0.20*pl.col("feat_ranking_depth") + 0.15*pl.col("feat_retrieval_depth") + 0.15*pl.col("feat_evaluation_rigor") + 0.35*pl.col("feat_builder_score"))).alias("Technical_Multiplier")
        )
    if 'contradiction_score' in mod_dict:
        df_sim = df_sim.with_columns(((-0.10 * pl.col("contradiction_score")).exp()).alias("Honeypot_Decay"))
        
    df_sim = df_sim.with_columns(
        (pl.col("Base_Score") * pl.col("Technical_Multiplier") * pl.col("feat_verified_search_skill") * pl.col("Trajectory_Multiplier") * pl.col("Behavioral_Multiplier") * pl.col("Persona_Penalty") * pl.col("Honeypot_Decay")).alias("Final_Score")
    )
    df_sim_ranked = df_sim.sort(["Final_Score","candidate_id"], descending=[True,False]).with_row_index("sim_rank")
    return df_sim_ranked.filter(pl.col("candidate_id") == cid)['sim_rank'][0] + 1

print("\n" + "="*80)
print("SCORE DECOMPOSITION")
print("="*80)

for cid in target_candidates:
    row = df_ranked.filter(pl.col("candidate_id") == cid)
    if len(row) == 0: continue
    r = row[0]
    rank = r['rank'][0] + 1
    
    base = r['Base_Score'][0]
    builder = r['feat_builder_score'][0]
    sb = r['feat_search_builder'][0]
    ret = r['feat_retrieval_depth'][0]
    rank_depth = r['feat_ranking_depth'][0]
    ev = r['feat_evaluation_rigor'][0]
    cont = r['contradiction_score'][0]
    beh = r['Behavioral_Multiplier'][0]
    
    traj = r['Trajectory_Multiplier'][0]
    tech = r['Technical_Multiplier'][0]
    hp = r['Honeypot_Decay'][0]
    fs = r['Final_Score'][0]
    
    print(f"\nCandidate: {cid} (Current Rank: {rank})")
    print(f"  Base Semantic Score:   {base:.3f} (Top20 Avg: {t20_base:.3f})")
    print(f"  Builder Score:         {builder:.3f} (Top20 Avg: {t20_builder:.3f})")
    print(f"  Search Builder:        {sb:.3f} (Top20 Avg: {t20_sb:.3f})")
    print(f"  Retrieval Depth:       {ret:.3f} (Top20 Avg: {t20_ret:.3f})")
    print(f"  Ranking Depth:         {rank_depth:.3f} (Top20 Avg: {t20_rank:.3f})")
    print(f"  Evaluation Rigor:      {ev:.3f} (Top20 Avg: {t20_eval:.3f})")
    print(f"  Contradiction Score:   {cont:.1f} (Top20 Avg: {t20_cont:.1f})")
    print(f"  Behavioral Score:      {beh:.3f} (Top20 Avg: {t20_beh:.3f})")
    print(f"  --- Multipliers ---")
    print(f"  Technical: {tech:.3f} | Trajectory: {traj:.3f} | Honeypot: {hp:.3f} | Behavioral: {beh:.3f}")
    print(f"  FINAL SCORE: {fs:.4f}")
    
    # Identify bottlenecks
    bottlenecks = []
    if base < t20_base - 0.05: bottlenecks.append(("Base_Score", t20_base))
    if builder < t20_builder - 0.2: bottlenecks.append(("feat_builder_score", t20_builder))
    if ret < t20_ret - 0.2: bottlenecks.append(("feat_retrieval_depth", t20_ret))
    if rank_depth < t20_rank - 0.2: bottlenecks.append(("feat_ranking_depth", t20_rank))
    if cont > 0: bottlenecks.append(("contradiction_score", 0.0))
    if beh < t20_beh - 0.05: bottlenecks.append(("Behavioral_Multiplier", t20_beh))
    
    print("  --- Simulations ---")
    if not bottlenecks:
        print("  No major bottlenecks relative to Top 20.")
    for feat, target_val in bottlenecks:
        new_rank = get_simulated_rank(cid, {feat: target_val})
        gain = rank - new_rank
        print(f"  If {feat} was {target_val:.3f} -> Rank {new_rank} (Gain: {gain})")

print("\n" + "="*80)
print("ROOT CAUSE TABLE DATA GENERATION COMPLETE")
print("="*80)
