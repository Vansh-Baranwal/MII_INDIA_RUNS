import polars as pl
import numpy as np
import faiss
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

artifacts_dir = Path(r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS\artifacts")

# Read pool and features
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

df_parsed = pl.read_parquet(artifacts_dir / 'parsed_candidates.parquet', glob=False)
df_pool = df_pool.join(df_parsed, on="candidate_id", how="left")
if 'years_of_experience_right' in df_pool.columns:
    df_pool = df_pool.drop('years_of_experience_right')

df_pool = df_pool.with_columns([
    (0.55*pl.col("sim_recent") + 0.30*pl.col("sim_last_two") + 0.15*pl.col("sim_full")).alias("Base_Score"),
    ((0.4 * pl.col("feat_builder_score") + 0.3 * pl.col("feat_ranking_depth") + 0.3 * pl.col("feat_retrieval_depth")).clip(0.0, 1.0)).alias("feat_search_builder"),
    (1.0 + (pl.col("feat_availability_score") + pl.col("feat_saved_boost") + pl.col("feat_search_appearance_boost") - 1.0)*0.25).alias("Behavioral_Multiplier"),
    (pl.col("feat_wrapper_ai_only") * pl.col("feat_architect_no_coding")).alias("Persona_Penalty")
])

def compute_raw_contradiction(row):
    score = 0.0
    yoe = row.get('years_of_experience', 0)
    if yoe is None: yoe = 0
    grad_year = row.get('grad_year', 0)
    if grad_year is None: grad_year = 0
            
    if grad_year > 0:
        if (2026 - grad_year) < yoe - 2:
            score += 5.0
            
    sal_min = row.get('expected_salary_min', 0)
    if sal_min is None: sal_min = 0
    if yoe <= 2 and sal_min > 40:
        score += 5.0
        
    views = row.get('profile_views_received_30d', 0)
    if views is None: views = 0
    response_rate = row.get('recruiter_response_rate', 0.0)
    if response_rate is None: response_rate = 0.0
    if response_rate == 1.0 and views == 0:
        score += 2.0
        
    try:
        skills = json.loads(row.get('skills_json', '[]'))
        for s in skills:
            if s.get('proficiency') == 'expert' and s.get('duration_months', 0) == 0:
                score += 1.0
    except:
        pass
    return score

df_pool = df_pool.with_columns(pl.Series("raw_contra", [compute_raw_contradiction(r) for r in df_pool.iter_rows(named=True)]))

df_pool = df_pool.with_columns(
    pl.when(
        (pl.col("feat_search_builder") >= 0.80) | 
        ((pl.col("feat_ranking_depth") >= 0.80) & (pl.col("feat_retrieval_depth") >= 0.80))
    )
    .then(pl.col("raw_contra") * 0.25)
    .otherwise(pl.col("raw_contra"))
    .alias("exempt_contra")
)

df_pool = df_pool.with_columns([
    ((-0.10 * pl.col("exempt_contra")).exp()).alias("Honeypot_Decay"),
    (0.50 + 0.20 * pl.col("feat_builder_score") + 0.20 * pl.col("feat_ranking_depth") + 0.10 * pl.col("feat_retrieval_depth")).alias("Trajectory_Multiplier"),
    (1.0 + (0.25*pl.col("feat_search_relevance_evidence") + 0.20*pl.col("feat_ranking_depth") + 0.15*pl.col("feat_retrieval_depth") + 0.15*pl.col("feat_evaluation_rigor") + 0.35*pl.col("feat_builder_score"))).alias("Technical_Multiplier")
])

df_pool = df_pool.with_columns(
    (pl.col("Technical_Multiplier") * pl.col("feat_verified_search_skill") * pl.col("Trajectory_Multiplier") * pl.col("Behavioral_Multiplier") * pl.col("Persona_Penalty") * pl.col("Honeypot_Decay")).alias("Core_Mult")
)

# Variants
variants = {
    "Variant A (Square Root)": (pl.col("Base_Score")**0.5) * pl.col("Core_Mult"),
    "Variant B (0.5 + 0.5*BaseScore)": (0.5 + 0.5 * pl.col("Base_Score")) * pl.col("Core_Mult"),
    "Variant C (BaseScore as Tiebreaker)": pl.col("Core_Mult"), # Handled by sorting logic
    "Variant D (1 + 0.20*BaseScore)": pl.col("Core_Mult") * (1.0 + 0.20 * pl.col("Base_Score")),
}

print("\n" + "="*120)
print("BASE SCORE ARCHITECTURE EXPERIMENTS")
print("="*120)

targets = ["CAND_0092278", "CAND_0030953", "CAND_0018499", "CAND_0083307"]

for v_name, expr in variants.items():
    df_pool = df_pool.with_columns(expr.alias(f"Final_{v_name}"))
    
    if v_name == "Variant C (BaseScore as Tiebreaker)":
        df_ranked = df_pool.sort([f"Final_{v_name}", "Base_Score", "candidate_id"], descending=[True, True, False]).with_row_index("rank")
    else:
        df_ranked = df_pool.sort([f"Final_{v_name}", "candidate_id"], descending=[True, False]).with_row_index("rank")
        
    top20 = df_ranked.head(20)
    top100 = df_ranked.head(100)
    
    b_mean = top100['feat_builder_score'].mean()
    r_mean = top100['feat_retrieval_depth'].mean()
    rd_mean = top100['feat_ranking_depth'].mean()
    
    titles100 = top100['current_title'].fill_null("").to_list()
    junior = sum(1 for t in titles100 if 'junior' in t.lower() or 'associate' in t.lower())
    cv = sum(1 for t in titles100 if any(k in t.lower() for k in ['computer vision', 'vision']))
    
    print(f"\n{v_name}")
    print(f"  Top 100 Averages:")
    print(f"    Builder: {b_mean:.3f} | Retrieval: {r_mean:.3f} | Ranking: {rd_mean:.3f}")
    print(f"    Contamination: {junior} Juniors, {cv} CV Engineers")
    
    print("\n  Target Movement:")
    for cid in targets:
        row = df_ranked.filter(pl.col("candidate_id") == cid)
        if len(row) > 0:
            rank = row['rank'][0] + 1
            print(f"    {cid:<15} | Rank: {rank:>4}")
            
    print("\n  Top 20:")
    for row in top20.iter_rows(named=True):
        print(f"    {row['rank']+1:>2} | {row['candidate_id']} | {row['current_title']:<35} | Final={row[f'Final_{v_name}']:.4f}")
    print("-" * 120)

print("\nDone.")
