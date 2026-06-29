import polars as pl
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import scipy.stats

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
df_pool = df_pool.with_columns(
    Base_Score=(0.55*pl.col("sim_recent") + 0.30*pl.col("sim_last_two") + 0.15*pl.col("sim_full")),
    Behavioral_Multiplier=1.0 + (pl.col("feat_availability_score") + pl.col("feat_saved_boost") + pl.col("feat_search_appearance_boost") - 1.0)*0.25,
    Trajectory_Multiplier=pl.col("feat_product_exposure") + pl.col("feat_trajectory_transition"),
    Persona_Penalty=pl.col("feat_wrapper_ai_only") * pl.col("feat_architect_no_coding"),
    Honeypot_Decay=(-pl.col("contradiction_score")).exp(),
)
df_pool = df_pool.with_columns(
    Technical_Multiplier=1.0 + (0.25*pl.col("feat_search_relevance_evidence") + 0.20*pl.col("feat_ranking_depth") + 0.15*pl.col("feat_retrieval_depth") + 0.15*pl.col("feat_evaluation_rigor") + 0.35*pl.col("feat_builder_score")),
)
df_pool = df_pool.with_columns(
    Final_Score=(pl.col("Base_Score") * pl.col("Technical_Multiplier") * pl.col("feat_verified_search_skill") * pl.col("Trajectory_Multiplier") * pl.col("Behavioral_Multiplier") * pl.col("Persona_Penalty") * pl.col("Honeypot_Decay")),
)
df_ranked = df_pool.sort(["Final_Score","candidate_id"], descending=[True,False]).with_row_index("rank")

# Add clean_ranking_depth to compute Elite41
df_all = pl.read_parquet(artifacts_dir / 'parsed_candidates.parquet', glob=False)
pool_cids = set(df_ranked['candidate_id'].to_list())

# TIER1, TIER2, TIER3
NEW_DICT = {"learning-to-rank": 1.0, "learning to rank": 1.0, "lambdarank": 1.0, "lambdamart": 1.0, "ranknet": 1.0,
            "pairwise ranking": 0.9, "listwise ranking": 0.9, "pointwise ranking": 0.9, "click model": 0.9,
            "ctr prediction": 0.8, "click-through rate prediction": 0.8, "position bias": 0.9, "ndcg": 0.8,
            "mrr": 0.8, "dcg": 0.8, "ranking evaluation": 0.8,
            "re-ranking": 0.5, "reranking": 0.5, "search ranking": 0.5, "ranking system": 0.5,
            "ranking pipeline": 0.5, "multi-stage ranking": 0.6, "two-stage retrieval": 0.6,
            "search relevance": 0.4, "relevance model": 0.5, "recall stage": 0.4, "precision stage": 0.4,
            "query ranking": 0.5, "candidate ranking": 0.4, "retrieval ranking": 0.4, "map@k": 0.6}

clean_rd_map = {}
for row in df_all.iter_rows(named=True):
    cid = row['candidate_id']
    if cid not in pool_cids: continue
    txt = row['full_profile_text'].lower()
    raw = sum(txt.count(term) * w for term, w in NEW_DICT.items())
    clean_rd_map[cid] = min(raw / 3.0, 1.0)

df_ranked = df_ranked.with_columns(clean_ranking_depth=pl.Series([clean_rd_map.get(cid, 0.0) for cid in df_ranked['candidate_id']]))

# Load parsed dataframe for current_title
df_ranked = df_ranked.join(df_all.select(["candidate_id", "current_title"]), on="candidate_id", how="left")


# Top 100 Output
print(f"\n{'='*100}")
print("TOP 100 CANDIDATES")
print(f"{'='*100}")
print(f"{'ID':<16} {'Title':<35} {'ProdExp':>8} {'Builder':>8} {'RetD':>8} {'RankD':>8} {'YoE':>6}")
print("-"*100)

top100 = df_ranked.head(100)
for row in top100.iter_rows(named=True):
    print(f"{row['candidate_id']:<16} {str(row.get('current_title', '')):<35} {row['feat_product_exposure']:8.3f} {row['feat_builder_score']:8.3f} {row['feat_retrieval_depth']:8.3f} {row['feat_ranking_depth']:8.3f} {row['years_of_experience']:6.1f}")

print("\n\n")

print(f"\n{'='*100}")
print("A. How is feat_product_exposure calculated exactly?")
print(f"{'='*100}")
print("""
Initial Base Score: 0.50

Positive Rules (Additive):
+0.20 IF company_size in ['1-10', '11-50', '51-200']
+0.20 IF industry in ['software', 'internet', 'saas', 'consumer']
+0.30 IF job description contains any: ['saas', 'our product', 'b2b', 'b2c', 'startup', 'scale-up']

Negative Rules (Subtractive):
-0.50 IF company name matches any IT Services firm: ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "hcl", "deloitte"]
-0.30 IF job description contains any: ['client', 'delivery', 'sow', 'consulting', 'offshore']

Final Operation:
Score is mathematically clamped between [0.0, 1.0]
""")

# B. Distribution audit
top20 = df_ranked.head(20)
elite41 = df_ranked.filter(pl.col("clean_ranking_depth") >= 0.90)

print(f"\n{'='*100}")
print("B. Distribution Audit (feat_product_exposure)")
print(f"{'='*100}")
print(f"  Top 20 Mean:      {top20['feat_product_exposure'].mean():.3f}")
print(f"  Top 100 Mean:     {top100['feat_product_exposure'].mean():.3f}")
print(f"  Elite 41 Mean:    {elite41['feat_product_exposure'].mean():.3f}")
print(f"  Full Pool Mean:   {df_ranked['feat_product_exposure'].mean():.3f}")

# C. Correlation audit
print(f"\n{'='*100}")
print("C. Correlation Audit (feat_product_exposure vs other metrics)")
print(f"{'='*100}")
prod_vals = df_ranked['feat_product_exposure'].to_numpy()
for col in ['feat_builder_score', 'feat_retrieval_depth', 'feat_ranking_depth', 'Final_Score', 'clean_ranking_depth']:
    vals = df_ranked[col].to_numpy()
    corr, p = scipy.stats.pearsonr(prod_vals, vals)
    print(f"  Correlation with {col:<25}: {corr:6.3f} (p={p:.2e})")

# D. False Positive Audit
print(f"\n{'='*100}")
print("D. False-Positive Audit: Top 25 Candidates by feat_product_exposure")
print(f"{'='*100}")

top_prod = df_ranked.sort("feat_product_exposure", descending=True).head(25)
for row in top_prod.iter_rows(named=True):
    print(f"  {row['candidate_id']:<16} {str(row.get('current_title', '')):<35} | ProdExp={row['feat_product_exposure']:.3f} | Rank={row['rank']+1}")

print("""
Explanation:
Any candidate working at a small software startup (company_size < 200, industry=software) and using words like "our product" or "b2b" automatically hits 1.0 (0.5 + 0.2 + 0.2 + 0.3 = 1.2, clamped to 1.0).
This assigns maximum possible Trajectory Multiplier (Trajectory = ProductExposure + TrajectoryTransition).

Would a Redrob judge consider that evidence of search/ranking ability?
NO. Working at a SaaS startup does not mean you know how to build a Learning-to-Rank system or optimize NDCG. It is purely a proxy for employer type, completely disconnected from technical skill.
""")

# E. Buried Elite Audit
buried_ids = ["CAND_0092278", "CAND_0077337", "CAND_0060072", "CAND_0083307", "CAND_0018499"]
print(f"\n{'='*100}")
print("E. Buried Elite Audit")
print(f"{'='*100}")

for cid in buried_ids:
    row = df_ranked.filter(pl.col("candidate_id") == cid).row(0, named=True)
    print(f"Candidate: {cid}")
    print(f"  Title: {row.get('current_title')}")
    print(f"  Rank:  {row['rank']+1}")
    print(f"  Final Score: {row['Final_Score']:.4f}")
    print("  Decomposition:")
    print(f"    Base_Score:           {row['Base_Score']:.3f}")
    print(f"    Technical_Multiplier: {row['Technical_Multiplier']:.3f}")
    print(f"      - Builder:       {row['feat_builder_score']:.3f}")
    print(f"      - Retrieval:     {row['feat_retrieval_depth']:.3f}")
    print(f"      - Ranking:       {row['feat_ranking_depth']:.3f} (Clean: {row['clean_ranking_depth']:.3f})")
    print(f"      - SearchRel:     {row['feat_search_relevance_evidence']:.3f}")
    print(f"    Trajectory_Multiplier: {row['Trajectory_Multiplier']:.3f}")
    print(f"      - Product Exposure:  {row['feat_product_exposure']:.3f}   <--- SUPPRESSION POINT")
    print(f"      - Traj Transition:   {row['feat_trajectory_transition']:.3f}")
    print(f"    Behavioral_Multiplier: {row['Behavioral_Multiplier']:.3f}")
    print("-" * 50)
    
print("\nDone.")
