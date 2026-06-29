import polars as pl
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from collections import Counter

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

# Add clean_ranking_depth
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

# Baseline multipliers
df_pool = df_pool.with_columns([
    (0.55*pl.col("sim_recent") + 0.30*pl.col("sim_last_two") + 0.15*pl.col("sim_full")).alias("Base_Score"),
    (1.0 + (pl.col("feat_availability_score") + pl.col("feat_saved_boost") + pl.col("feat_search_appearance_boost") - 1.0)*0.25).alias("Behavioral_Multiplier"),
    (pl.col("feat_wrapper_ai_only") * pl.col("feat_architect_no_coding")).alias("Persona_Penalty"),
    ((-pl.col("contradiction_score")).exp()).alias("Honeypot_Decay"),
    (0.50 + 0.50 * pl.col("feat_builder_score")).alias("Trajectory_Multiplier")
])

# Helper for parsing titles
def get_title_stats(titles):
    search = sum(1 for t in titles if any(k in t.lower() for k in ['search', 'ranking', 'relevance', 'ir engineer', 'information retrieval']))
    recsys = sum(1 for t in titles if any(k in t.lower() for k in ['recommendation', 'recsys']))
    nlp = sum(1 for t in titles if any(k in t.lower() for k in ['nlp', 'natural language']))
    aml = sum(1 for t in titles if any(k in t.lower() for k in ['applied ml', 'applied machine learning', 'applied scientist']))
    cv = sum(1 for t in titles if any(k in t.lower() for k in ['computer vision', 'vision']))
    junior = sum(1 for t in titles if 'junior' in t.lower() or 'associate' in t.lower())
    research = sum(1 for t in titles if 'research' in t.lower() or 'scientist' in t.lower() and 'data' not in t.lower() and 'applied' not in t.lower())
    return search, recsys, nlp, aml, junior, research, cv

target_candidates = ["CAND_0024466", "CAND_0018549", "CAND_0092278", "CAND_0077337"]

def simulate_ranking_weight(name, rank_wt):
    # tech_mult = 1.0 + (0.25*SR + rank_wt*clean_RD + 0.15*RetD + 0.15*Eval + 0.35*Builder)
    df_sim = df_pool.with_columns(
        (1.0 + (0.25*pl.col("feat_search_relevance_evidence") + rank_wt*pl.col("clean_ranking_depth") + 0.15*pl.col("feat_retrieval_depth") + 0.15*pl.col("feat_evaluation_rigor") + 0.35*pl.col("feat_builder_score"))).alias("Technical_Multiplier")
    )
    df_sim = df_sim.with_columns(
        (pl.col("Base_Score") * pl.col("Technical_Multiplier") * pl.col("feat_verified_search_skill") * pl.col("Trajectory_Multiplier") * pl.col("Behavioral_Multiplier") * pl.col("Persona_Penalty") * pl.col("Honeypot_Decay")).alias("Final_Score")
    )
    df_ranked = df_sim.sort(["Final_Score","candidate_id"], descending=[True,False]).with_row_index("rank")
    
    top20 = df_ranked.head(20)
    top100 = df_ranked.head(100)
    
    # 3-6. Avgs for Top 100
    b_mean = top100['feat_builder_score'].mean()
    r_mean = top100['feat_retrieval_depth'].mean()
    rd_mean = top100['clean_ranking_depth'].mean()
    yoe_mean = top100['years_of_experience'].mean()
    
    titles100 = top100['current_title'].fill_null("").to_list()
    s_c, r_c, n_c, aml_c, j_c, res_c, cv_c = get_title_stats(titles100)
    
    print(f"\n{'='*100}")
    print(f"  {name}  (RankingDepth Weight: {rank_wt:.2f})")
    print(f"{'='*100}")
    print(f"  Avg Builder Score:    {b_mean:.3f}")
    print(f"  Avg Retrieval Depth:  {r_mean:.3f}")
    print(f"  Avg Ranking Depth:    {rd_mean:.3f}")
    print(f"  Avg Years Experience: {yoe_mean:.1f}")
    print(f"  Titles (Top 100):")
    print(f"    Search/Ranking:   {s_c}")
    print(f"    Recommendation:   {r_c}")
    print(f"    NLP:              {n_c}")
    print(f"    Applied ML:       {aml_c}")
    print(f"    Junior:           {j_c}")
    print(f"    Research:         {res_c}")
    print(f"    Computer Vis:     {cv_c}")
    
    print(f"\n  Target Candidates:")
    for cid in target_candidates:
        row = df_ranked.filter(pl.col("candidate_id") == cid)
        if len(row) > 0:
            rank = row['rank'][0] + 1
            print(f"    {cid:<15} Rank: {rank:>4}")
        else:
            print(f"    {cid:<15} Rank:  N/A")
            
    print(f"\n  Top 20 Candidates:")
    for row in top20.iter_rows(named=True):
        print(f"    {row['rank']+1:>2} | {row['candidate_id']} | {row['current_title']:<35} | B={row['feat_builder_score']:.2f} | RD={row['clean_ranking_depth']:.2f}")

scenarios = [
    ("Scenario A (Current)", 0.20),
    ("Scenario B (+25%)", 0.25),
    ("Scenario C (+50%)", 0.30),
    ("Scenario D (+75%)", 0.35),
    ("Scenario E (+100%)", 0.40),
]

for name, wt in scenarios:
    simulate_ranking_weight(name, wt)

print("\nDone.")
