import polars as pl
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import copy

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

df_parsed = pl.read_parquet(artifacts_dir / 'parsed_candidates.parquet', glob=False).select(["candidate_id", "current_title", "years_of_experience", "full_profile_text"])
df_pool = df_pool.join(df_parsed, on="candidate_id", how="left")
if 'years_of_experience_right' in df_pool.columns:
    df_pool = df_pool.drop('years_of_experience_right')

# Clean ranking depth
NEW_DICT = {"learning-to-rank": 1.0, "learning to rank": 1.0, "lambdarank": 1.0, "lambdamart": 1.0, "ranknet": 1.0,
            "pairwise ranking": 0.9, "listwise ranking": 0.9, "pointwise ranking": 0.9, "click model": 0.9,
            "ctr prediction": 0.8, "click-through rate prediction": 0.8, "position bias": 0.9, "ndcg": 0.8,
            "mrr": 0.8, "dcg": 0.8, "ranking evaluation": 0.8,
            "re-ranking": 0.5, "reranking": 0.5, "search ranking": 0.5, "ranking system": 0.5,
            "ranking pipeline": 0.5, "multi-stage ranking": 0.6, "two-stage retrieval": 0.6,
            "search relevance": 0.4, "relevance model": 0.5, "recall stage": 0.4, "precision stage": 0.4,
            "query ranking": 0.5, "candidate ranking": 0.4, "retrieval ranking": 0.4, "map@k": 0.6}

pool_cids = set(df_pool['candidate_id'].to_list())
clean_rd_map = {}
for row in df_parsed.iter_rows(named=True):
    cid = row['candidate_id']
    if cid not in pool_cids: continue
    txt = row['full_profile_text'].lower()
    raw = sum(txt.count(term) * w for term, w in NEW_DICT.items())
    clean_rd_map[cid] = min(raw / 3.0, 1.0)

df_pool = df_pool.with_columns(clean_ranking_depth=pl.Series([clean_rd_map.get(cid, 0.0) for cid in df_pool['candidate_id']]))

# Validated Scenario E baseline
df_pool = df_pool.with_columns([
    (0.55*pl.col("sim_recent") + 0.30*pl.col("sim_last_two") + 0.15*pl.col("sim_full")).alias("Base_Score"),
    (1.0 + (pl.col("feat_availability_score") + pl.col("feat_saved_boost") + pl.col("feat_search_appearance_boost") - 1.0)*0.25).alias("Behavioral_Multiplier"),
    (pl.col("feat_wrapper_ai_only") * pl.col("feat_architect_no_coding")).alias("Persona_Penalty"),
    ((-pl.col("contradiction_score")).exp()).alias("Honeypot_Decay"),
    (0.50 + 0.50 * pl.col("feat_builder_score")).alias("Trajectory_Multiplier"),
    (1.0 + (0.25*pl.col("feat_search_relevance_evidence") + 0.20*pl.col("clean_ranking_depth") + 0.15*pl.col("feat_retrieval_depth") + 0.15*pl.col("feat_evaluation_rigor") + 0.35*pl.col("feat_builder_score"))).alias("Technical_Multiplier")
])

df_sim = df_pool.with_columns(
    (pl.col("Base_Score") * pl.col("Technical_Multiplier") * pl.col("feat_verified_search_skill") * pl.col("Trajectory_Multiplier") * pl.col("Behavioral_Multiplier") * pl.col("Persona_Penalty") * pl.col("Honeypot_Decay")).alias("Final_Score")
)
df_ranked = df_sim.sort(["Final_Score","candidate_id"], descending=[True,False]).with_row_index("rank")

# 1. Top 100 by Builder
top100_builder = set(df_ranked.sort(["feat_builder_score","candidate_id"], descending=[True,False]).head(100)['candidate_id'].to_list())

# 2. Top 100 by Clean Ranking Depth
top100_rd = set(df_ranked.sort(["clean_ranking_depth","candidate_id"], descending=[True,False]).head(100)['candidate_id'].to_list())

# Union
elite_cids = top100_builder.union(top100_rd)

# Buried (> 500)
df_buried = df_ranked.filter((pl.col("candidate_id").is_in(list(elite_cids))) & (pl.col("rank") >= 500))

print(f"\n{'='*100}")
print(f"Buried Elite Candidates Audit (Rank >= 500, n={len(df_buried)})")
print(f"{'='*100}")

categories = {
    'contradiction penalty': 0,
    'low builder multiplier': 0,
    'low semantic score': 0,
    'low retrieval depth': 0,
    'behavioral damping': 0,
    'persona penalty': 0,
    'other': 0
}

candidate_reports = []

for row in df_buried.iter_rows(named=True):
    # Determine root cause
    cause = "other"
    if row['Honeypot_Decay'] < 1.0:
        cause = "contradiction penalty"
    elif row['Trajectory_Multiplier'] < 0.65:
        cause = "low builder multiplier"
    elif row['Base_Score'] < 0.40:
        cause = "low semantic score"
    elif row['feat_retrieval_depth'] < 0.20:
        cause = "low retrieval depth"
    elif row['Behavioral_Multiplier'] < 0.90:
        cause = "behavioral damping"
    elif row['Persona_Penalty'] < 1.0:
        cause = "persona penalty"
    else:
        # Fallback check
        if row['Base_Score'] < 0.45: cause = "low semantic score"
        elif row['feat_retrieval_depth'] < 0.30: cause = "low retrieval depth"
        
    categories[cause] += 1
    
    rep = {
        'candidate_id': row['candidate_id'],
        'current_rank': row['rank'] + 1,
        'title': row['current_title'],
        'builder': row['feat_builder_score'],
        'ranking': row['clean_ranking_depth'],
        'retrieval': row['feat_retrieval_depth'],
        'search_rel': row['feat_search_relevance_evidence'],
        'contradiction': row['contradiction_score'],
        'prod_exp': row['feat_product_exposure'],
        'traj_mult': row['Trajectory_Multiplier'],
        'base': row['Base_Score'],
        'tech': row['Technical_Multiplier'],
        'behav': row['Behavioral_Multiplier'],
        'final': row['Final_Score'],
        'cause': cause
    }
    candidate_reports.append(rep)

# Sort reports by rank
candidate_reports.sort(key=lambda x: x['current_rank'])

for r in candidate_reports[:20]: # Print top 20 buried for brevity
    print(f"\n{r['candidate_id']} | Rank: {r['current_rank']} | Title: {r['title']}")
    print(f"  Cause: {r['cause'].upper()}")
    print(f"  Builder: {r['builder']:.3f} | Ranking: {r['ranking']:.3f} | Retrieval: {r['retrieval']:.3f} | SearchRel: {r['search_rel']:.3f}")
    print(f"  Contra:  {r['contradiction']:.3f} | ProdExp: {r['prod_exp']:.3f}")
    print(f"  Decomp: Base={r['base']:.3f} * Tech={r['tech']:.3f} * Traj={r['traj_mult']:.3f} * Behav={r['behav']:.3f} -> Final={r['final']:.4f}")

if len(candidate_reports) > 20:
    print(f"\n... and {len(candidate_reports)-20} more candidates.")

print(f"\n{'='*100}")
print("Root Cause Analysis Summary")
print(f"{'='*100}")
sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
for i, (cat, count) in enumerate(sorted_cats):
    print(f"{i+1}. {cat:<25}: {count} candidates")

# --- Simulate Rank Improvement ---
# To estimate rank improvement, we need to temporarily "fix" each cause independently for all affected candidates, 
# compute their new scores, and see where they would rank in the original df_ranked.
print(f"\n{'='*100}")
print("Estimated Rank Improvement (Independent Fixes)")
print(f"{'='*100}")

orig_scores = np.array(df_ranked['Final_Score'].to_list())

for cause, count in sorted_cats:
    if count == 0: continue
    affected = [r for r in candidate_reports if r['cause'] == cause]
    
    improvements = []
    for r in affected:
        row = df_ranked.filter(pl.col("candidate_id") == r['candidate_id']).row(0, named=True)
        new_score = row['Final_Score']
        
        if cause == 'contradiction penalty':
            new_score = row['Base_Score'] * row['Technical_Multiplier'] * row['feat_verified_search_skill'] * row['Trajectory_Multiplier'] * row['Behavioral_Multiplier'] * row['Persona_Penalty'] * 1.0
        elif cause == 'low builder multiplier':
            new_traj = 0.5 + 0.5 * max(row['feat_builder_score'], 0.8) # simulate fixing the multiplier
            new_score = row['Base_Score'] * row['Technical_Multiplier'] * row['feat_verified_search_skill'] * new_traj * row['Behavioral_Multiplier'] * row['Persona_Penalty'] * row['Honeypot_Decay']
        elif cause == 'low semantic score':
            new_score = 0.50 * row['Technical_Multiplier'] * row['feat_verified_search_skill'] * row['Trajectory_Multiplier'] * row['Behavioral_Multiplier'] * row['Persona_Penalty'] * row['Honeypot_Decay']
        elif cause == 'low retrieval depth':
            new_tech = 1.0 + (0.25*row['feat_search_relevance_evidence'] + 0.20*row['clean_ranking_depth'] + 0.15*max(row['feat_retrieval_depth'], 0.8) + 0.15*row['feat_evaluation_rigor'] + 0.35*row['feat_builder_score'])
            new_score = row['Base_Score'] * new_tech * row['feat_verified_search_skill'] * row['Trajectory_Multiplier'] * row['Behavioral_Multiplier'] * row['Persona_Penalty'] * row['Honeypot_Decay']
        elif cause == 'behavioral damping':
            new_score = row['Base_Score'] * row['Technical_Multiplier'] * row['feat_verified_search_skill'] * row['Trajectory_Multiplier'] * 1.0 * row['Persona_Penalty'] * row['Honeypot_Decay']
            
        new_rank = np.searchsorted(-orig_scores, -new_score)
        improvements.append(r['current_rank'] - new_rank)
        
    print(f"  {cause:<25}: avg improvement of {np.mean(improvements):>5.1f} ranks (max: {np.max(improvements):>5})")

print("\nDone.")
