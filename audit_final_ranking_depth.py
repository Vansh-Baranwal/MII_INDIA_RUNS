import polars as pl
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from collections import Counter

artifacts_dir = Path(r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS\artifacts")

df_pool = pl.read_parquet(artifacts_dir / 'features.parquet', glob=False).with_row_index("faiss_id")
df_parsed = pl.read_parquet(artifacts_dir / 'parsed_candidates.parquet', glob=False).select(["candidate_id", "current_title", "years_of_experience", "full_profile_text"])

df_pool = df_pool.join(df_parsed, on="candidate_id", how="left")
if 'years_of_experience_right' in df_pool.columns:
    df_pool = df_pool.drop('years_of_experience_right')

OLD_DICT = {
    "learning-to-rank": 1.0, "lambdamart": 1.0, "re-ranking": 0.9, 
    "recommendation systems": 0.8, "personalization": 0.8, "collaborative filtering": 0.7,
    "xgboost": 0.7, "lightgbm": 0.7
}

pool_cids = set(df_pool['candidate_id'].to_list())
old_rd_map = {}
for row in df_parsed.iter_rows(named=True):
    cid = row['candidate_id']
    if cid not in pool_cids: continue
    txt = row['full_profile_text'].lower()
    raw = sum(txt.count(term) * w for term, w in OLD_DICT.items())
    old_rd_map[cid] = min(raw / 3.0, 1.0)

df_pool = df_pool.with_columns(old_ranking_depth=pl.Series([old_rd_map.get(cid, 0.0) for cid in df_pool['candidate_id']]))

print(f"\n{'='*100}")
print("ranking_depth Distribution Comparison (Entire Pool)")
print(f"{'='*100}")

old_elite = df_pool.filter(pl.col("old_ranking_depth") >= 0.90)
new_elite = df_pool.filter(pl.col("feat_ranking_depth") >= 0.90)

def analyze_elite_pool(name, group_df, target_col):
    count = len(group_df)
    if count == 0:
        print(f"  {name}: 0 candidates score >= 0.90")
        return
    b_mean = group_df['feat_builder_score'].mean()
    r_mean = group_df['feat_retrieval_depth'].mean()
    titles = group_df['current_title'].fill_null("").to_list()
    
    print(f"\n  {name} (Count: {count})")
    print(f"    Avg Builder Score:   {b_mean:.3f}")
    print(f"    Avg Retrieval Depth: {r_mean:.3f}")
    print(f"    Top Titles:")
    for t, c in Counter(titles).most_common(5):
        print(f"      {t}: {c}")

analyze_elite_pool("OLD ranking_depth >= 0.90", old_elite, "old_ranking_depth")
analyze_elite_pool("NEW ranking_depth >= 0.90", new_elite, "feat_ranking_depth")

# --- Ranking Recomputation ---
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

pool_cids_faiss = set(df_pool.filter(pl.col("faiss_id").is_in(list(indices_set)))['candidate_id'].to_list())
df_sub = df_pool.filter(pl.col("candidate_id").is_in(list(pool_cids_faiss)))

emb_recent = np.load(artifacts_dir / 'embeddings_recent.npy')
emb_last_two = np.load(artifacts_dir / 'embeddings_last_two.npy')
emb_full = np.load(artifacts_dir / 'embeddings_full.npy')
faiss.normalize_L2(emb_recent)
faiss.normalize_L2(emb_last_two)
faiss.normalize_L2(emb_full)

pool_ids = df_sub['faiss_id'].to_list()
df_sub = df_sub.with_columns([
    pl.Series("sim_recent", (emb_recent[pool_ids] @ jd_emb[0]).tolist()),
    pl.Series("sim_last_two", (emb_last_two[pool_ids] @ jd_emb[0]).tolist()),
    pl.Series("sim_full", (emb_full[pool_ids] @ jd_emb[0]).tolist()),
])

df_sub = df_sub.with_columns([
    (0.55*pl.col("sim_recent") + 0.30*pl.col("sim_last_two") + 0.15*pl.col("sim_full")).alias("Base_Score"),
    (1.0 + (pl.col("feat_availability_score") + pl.col("feat_saved_boost") + pl.col("feat_search_appearance_boost") - 1.0)*0.25).alias("Behavioral_Multiplier"),
    (pl.col("feat_wrapper_ai_only") * pl.col("feat_architect_no_coding")).alias("Persona_Penalty"),
    ((-0.10 * pl.col("contradiction_score")).exp()).alias("Honeypot_Decay"),
])

def compute_final_ranking(df, rd_col, name):
    df_sim = df.with_columns([
        (0.50 + 0.20 * pl.col("feat_builder_score") + 0.20 * pl.col(rd_col) + 0.10 * pl.col("feat_retrieval_depth")).alias("Trajectory_Multiplier"),
        (1.0 + (0.25*pl.col("feat_search_relevance_evidence") + 0.20*pl.col(rd_col) + 0.15*pl.col("feat_retrieval_depth") + 0.15*pl.col("feat_evaluation_rigor") + 0.35*pl.col("feat_builder_score"))).alias("Technical_Multiplier")
    ])
    df_sim = df_sim.with_columns(
        (pl.col("Base_Score") * pl.col("Technical_Multiplier") * pl.col("feat_verified_search_skill") * pl.col("Trajectory_Multiplier") * pl.col("Behavioral_Multiplier") * pl.col("Persona_Penalty") * pl.col("Honeypot_Decay")).alias("Final_Score")
    )
    df_ranked = df_sim.sort(["Final_Score","candidate_id"], descending=[True,False]).with_row_index("rank")
    
    top20 = df_ranked.head(20)
    top100 = df_ranked.head(100)
    
    b_mean = top100['feat_builder_score'].mean()
    r_mean = top100['feat_retrieval_depth'].mean()
    rd_mean = top100[rd_col].mean()
    
    titles100 = top100['current_title'].fill_null("").to_list()
    search = sum(1 for t in titles100 if any(k in t.lower() for k in ['search', 'ranking', 'relevance', 'ir engineer', 'information retrieval']))
    recsys = sum(1 for t in titles100 if any(k in t.lower() for k in ['recommendation', 'recsys']))
    nlp = sum(1 for t in titles100 if any(k in t.lower() for k in ['nlp', 'natural language']))
    aml = sum(1 for t in titles100 if any(k in t.lower() for k in ['applied ml', 'applied machine learning', 'applied scientist']))
    cv = sum(1 for t in titles100 if any(k in t.lower() for k in ['computer vision', 'vision']))
    junior = sum(1 for t in titles100 if 'junior' in t.lower() or 'associate' in t.lower())
    
    print(f"\n{'='*100}")
    print(f"  {name}")
    print(f"{'='*100}")
    print(f"  Avg Builder Score:    {b_mean:.3f}")
    print(f"  Avg Retrieval Depth:  {r_mean:.3f}")
    print(f"  Avg Ranking Depth:    {rd_mean:.3f}")
    
    print(f"\n  Top 100 Title Personas:")
    print(f"    Search/Ranking:   {search}")
    print(f"    Recommendation:   {recsys}")
    print(f"    NLP:              {nlp}")
    print(f"    Applied ML:       {aml}")
    print(f"    Junior ML:        {junior}")
    print(f"    Computer Vision:  {cv}")
            
    print(f"\n  Top 20 Candidates:")
    for row in top20.iter_rows(named=True):
        print(f"    {row['rank']+1:>2} | {row['candidate_id']} | {row['current_title']:<35} | Final={row['Final_Score']:.4f}")

compute_final_ranking(df_sub, "old_ranking_depth", "OLD Ranking Dictionary Baseline")
compute_final_ranking(df_sub, "feat_ranking_depth", "NEW Clean Ranking Dictionary")

print("\nDone.")
