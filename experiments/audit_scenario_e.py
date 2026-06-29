import polars as pl
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import json
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

# Common fields
df_pool = df_pool.with_columns([
    (0.55*pl.col("sim_recent") + 0.30*pl.col("sim_last_two") + 0.15*pl.col("sim_full")).alias("Base_Score"),
    (1.0 + (pl.col("feat_availability_score") + pl.col("feat_saved_boost") + pl.col("feat_search_appearance_boost") - 1.0)*0.25).alias("Behavioral_Multiplier"),
    (pl.col("feat_wrapper_ai_only") * pl.col("feat_architect_no_coding")).alias("Persona_Penalty"),
    ((-pl.col("contradiction_score")).exp()).alias("Honeypot_Decay"),
    (1.0 + (0.25*pl.col("feat_search_relevance_evidence") + 0.20*pl.col("feat_ranking_depth") + 0.15*pl.col("feat_retrieval_depth") + 0.15*pl.col("feat_evaluation_rigor") + 0.35*pl.col("feat_builder_score"))).alias("Technical_Multiplier")
])

# Scenario E:
df_pool = df_pool.with_columns(
    (0.50 + 0.50 * pl.col("feat_builder_score")).alias("Trajectory_Multiplier")
)

df_sim = df_pool.with_columns(
    (pl.col("Base_Score") * pl.col("Technical_Multiplier") * pl.col("feat_verified_search_skill") * pl.col("Trajectory_Multiplier") * pl.col("Behavioral_Multiplier") * pl.col("Persona_Penalty") * pl.col("Honeypot_Decay")).alias("Final_Score")
)
df_ranked = df_sim.sort(["Final_Score","candidate_id"], descending=[True,False]).with_row_index("rank")

top100 = df_ranked.head(100)

print(f"\n{'='*100}")
print("1. Title Distribution (Top 100)")
print(f"{'='*100}")
titles = top100['current_title'].fill_null("").to_list()
for t, c in Counter(titles).most_common():
    print(f"  {t}: {c}")

print(f"\n{'='*100}")
print("2. Years-of-Experience Distribution (Top 100)")
print(f"{'='*100}")
yoe = top100['years_of_experience'].to_numpy()
print(f"  Min: {np.min(yoe):.1f}")
print(f"  25th: {np.percentile(yoe, 25):.1f}")
print(f"  Median: {np.median(yoe):.1f}")
print(f"  Mean: {np.mean(yoe):.1f}")
print(f"  75th: {np.percentile(yoe, 75):.1f}")
print(f"  Max: {np.max(yoe):.1f}")

print(f"\n{'='*100}")
print("3. Builder Distribution (Top 100)")
print(f"{'='*100}")
b = top100['feat_builder_score'].to_numpy()
print(f"  Min: {np.min(b):.3f}")
print(f"  25th: {np.percentile(b, 25):.3f}")
print(f"  Median: {np.median(b):.3f}")
print(f"  Mean: {np.mean(b):.3f}")
print(f"  75th: {np.percentile(b, 75):.3f}")
print(f"  Max: {np.max(b):.3f}")

print(f"\n{'='*100}")
print("4. Retrieval Distribution (Top 100)")
print(f"{'='*100}")
r = top100['feat_retrieval_depth'].to_numpy()
print(f"  Min: {np.min(r):.3f}")
print(f"  25th: {np.percentile(r, 25):.3f}")
print(f"  Median: {np.median(r):.3f}")
print(f"  Mean: {np.mean(r):.3f}")
print(f"  75th: {np.percentile(r, 75):.3f}")
print(f"  Max: {np.max(r):.3f}")

print(f"\n{'='*100}")
print("5. Ranking Depth Distribution (Top 100)")
print(f"{'='*100}")
rd = top100['feat_ranking_depth'].to_numpy()
print(f"  Min: {np.min(rd):.3f}")
print(f"  25th: {np.percentile(rd, 25):.3f}")
print(f"  Median: {np.median(rd):.3f}")
print(f"  Mean: {np.mean(rd):.3f}")
print(f"  75th: {np.percentile(rd, 75):.3f}")
print(f"  Max: {np.max(rd):.3f}")

print(f"\n{'='*100}")
print("A. Top 20 Candidates with Builder > 0.80")
print(f"{'='*100}")
top20 = df_ranked.head(20)
high_b = top20.filter(pl.col("feat_builder_score") > 0.80)
for row in high_b.iter_rows(named=True):
    print(f"  {row['candidate_id']} | Rank {row['rank']+1:>2} | B={row['feat_builder_score']:.3f} | {row['current_title']}")

print(f"\n{'='*100}")
print("B. Top 20 Candidates with Builder < 0.30")
print(f"{'='*100}")
low_b = top20.filter(pl.col("feat_builder_score") < 0.30)
for row in low_b.iter_rows(named=True):
    print(f"  {row['candidate_id']} | Rank {row['rank']+1:>2} | B={row['feat_builder_score']:.3f} | {row['current_title']}")

# Persona filtering
search_kw = ['search engineer', 'search ranking', 'information retrieval', 'ir engineer', 'relevance engineer']
recsys_kw = ['recommendation', 'recsys', 'rec sys']
nlp_kw = ['nlp', 'natural language']

def get_persona(title):
    t = title.lower()
    if any(k in t for k in search_kw) or any(k in t for k in ['search', 'ranking', 'relevance']):
        return 'Search'
    elif any(k in t for k in recsys_kw):
        return 'Recommendation'
    elif any(k in t for k in nlp_kw):
        return 'NLP'
    return None

df_rest = df_ranked.filter(pl.col("rank") >= 100)
search_out = []
rec_out = []
nlp_out = []

for row in df_rest.iter_rows(named=True):
    p = get_persona(row['current_title'])
    if p == 'Search': search_out.append(row)
    elif p == 'Recommendation': rec_out.append(row)
    elif p == 'NLP': nlp_out.append(row)

def print_out_group(name, group):
    print(f"\n{'='*100}")
    print(f"{name} outside Top 100 (Count: {len(group)})")
    print(f"{'='*100}")
    print(f"{'ID':<16} {'Title':<35} {'Rank':>6} {'Builder':>8} {'RetD':>8} {'RankD':>8} {'Contradiction':>13}")
    print("-"*100)
    # Sort by rank and print top 20 of them
    for row in group[:20]:
        print(f"{row['candidate_id']:<16} {row['current_title']:<35} {row['rank']+1:6} {row['feat_builder_score']:8.3f} {row['feat_retrieval_depth']:8.3f} {row['feat_ranking_depth']:8.3f} {row['contradiction_score']:13.3f}")

print_out_group("C. Search Engineers", search_out)
print_out_group("D. Recommendation Engineers", rec_out)
print_out_group("E. NLP Engineers", nlp_out)

print("\nDone.")
