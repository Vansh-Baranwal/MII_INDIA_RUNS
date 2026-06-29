import polars as pl
from pathlib import Path
import json
import math

base_dir = Path(r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS")
artifacts_dir = base_dir / 'artifacts'

df_new_20 = pl.read_csv(base_dir / 'top20.csv', glob=False)
df_new_100 = pl.read_csv(base_dir / 'top100.csv', glob=False)

df_feat = pl.read_parquet(artifacts_dir / 'features.parquet', glob=False)
df_parsed = pl.read_parquet(artifacts_dir / 'parsed_candidates.parquet', glob=False)

# Recreate the old baseline to get old Top 20 / Top 100
# Old baseline: 
# dur_total was active. No Elite Exemption. BaseScore was multiplicative.
# Trajectory was: feat_product_exposure + feat_trajectory_transition
# Technical: 1.0 + 0.25*SR + 0.20*RD + 0.15*RetD + 0.15*Eval + 0.35*B

# I will just load the ORIGINAL parsed data (wait, we didn't change parsed_candidates)
df_pool = df_feat.join(df_parsed, on="candidate_id", how="inner")

def compute_old_contra(row):
    score = 0.0
    yoe = row.get('years_of_experience', 0)
    if yoe is None: yoe = 0
    dur_total = row.get('total_duration_months', 0)
    if dur_total is None: dur_total = 0
    grad_year = row.get('grad_year', 0)
    if grad_year is None: grad_year = 0
            
    if dur_total > (yoe * 12) * 1.2:
        score += 5.0
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

df_pool = df_pool.with_columns(pl.Series("old_contra", [compute_old_contra(r) for r in df_pool.iter_rows(named=True)]))

# For old base_score we need the exact embeddings score. Let's join from df_new_100?
# Actually, the base score hasn't changed.
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
jd_text = "Senior AI Engineer Search Ranking Retrieval Embeddings NDCG Vector Databases"
model = SentenceTransformer('all-MiniLM-L6-v2')
jd_emb = model.encode([jd_text])
faiss.normalize_L2(jd_emb)

emb_full = np.load(artifacts_dir / 'embeddings_full.npy')
faiss.normalize_L2(emb_full)

df_pool = df_pool.with_row_index("row_nr")
pool_ids = df_pool['faiss_id'].to_list()
sim_full = (emb_full[pool_ids] @ jd_emb[0]).tolist()
# wait, I don't have faiss_id easily aligned without indexer. 
# Just run a simple diff against what we printed earlier!
