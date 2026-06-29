import polars as pl
import numpy as np
import math
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from collections import Counter
from scipy import stats

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

pool_size = len(df_ranked)
print(f"Total pool: {pool_size}")

# --- Load full text to compute clean ranking depth ---
df_all = pl.read_parquet(artifacts_dir / 'parsed_candidates.parquet', glob=False)

# Clean dictionary (from previous audit)
TIER1 = {"learning-to-rank": 1.0, "learning to rank": 1.0, "lambdarank": 1.0, "lambdamart": 1.0, "ranknet": 1.0}
TIER2 = {"pairwise ranking": 0.9, "listwise ranking": 0.9, "pointwise ranking": 0.9, "click model": 0.9, "ctr prediction": 0.8, "click-through rate prediction": 0.8, "position bias": 0.9, "ndcg": 0.8, "mrr": 0.8, "dcg": 0.8, "ranking evaluation": 0.8}
TIER3 = {"re-ranking": 0.5, "reranking": 0.5, "search ranking": 0.5, "ranking system": 0.5, "ranking pipeline": 0.5, "multi-stage ranking": 0.6, "two-stage retrieval": 0.6, "search relevance": 0.4, "relevance model": 0.5, "recall stage": 0.4, "precision stage": 0.4, "query ranking": 0.5, "candidate ranking": 0.4, "retrieval ranking": 0.4, "map@k": 0.6}
NEW_DICT = {}
NEW_DICT.update(TIER1)
NEW_DICT.update(TIER2)
NEW_DICT.update(TIER3)

# Compute clean_ranking_depth for every pool candidate
pool_cids = set(df_ranked['candidate_id'].to_list())
clean_rd_map = {}

for row in df_all.iter_rows(named=True):
    cid = row['candidate_id']
    if cid not in pool_cids:
        continue
    txt = row['full_profile_text'].lower()
    raw = 0.0
    for term, w in NEW_DICT.items():
        raw += txt.count(term) * w
    clean_rd_map[cid] = min(raw / 3.0, 1.0)

# Add clean_ranking_depth to ranked df
crd_vals = [clean_rd_map.get(cid, 0.0) for cid in df_ranked['candidate_id'].to_list()]
df_ranked = df_ranked.with_columns(clean_ranking_depth=pl.Series(crd_vals))

# Binary feature
elite_vals = [1 if v >= 0.90 else 0 for v in crd_vals]
df_ranked = df_ranked.with_columns(elite_ranking_specialist=pl.Series(elite_vals))

# --- Section 1: Output the 41 elite candidates ---
elite41 = df_ranked.filter(pl.col("elite_ranking_specialist") == 1)

print(f"\n{'='*140}")
print(f"  41 Elite Ranking Specialists (clean_ranking_depth >= 0.90)")
print(f"{'='*140}")
print(f"{'ID':<16} {'Title':<35} {'YoE':>5} {'Builder':>8} {'RetD':>7} {'SR':>7} {'ProdExp':>8} {'Rank':>6}")
print("-"*100)

for row in elite41.sort("rank").iter_rows(named=True):
    print(f"{row['candidate_id']:<16} {str(row.get('current_title','')):<35} {row['years_of_experience']:5.1f} {row['feat_builder_score']:8.3f} {row['feat_retrieval_depth']:7.3f} {row['feat_search_relevance_evidence']:7.3f} {row['feat_product_exposure']:8.3f} {row['rank']+1:6}")

# --- Section A: Title distribution ---
titles = elite41['current_title'].fill_null("Unknown").to_list()
print(f"\n{'='*70}")
print("A) Title Distribution")
print(f"{'='*70}")
for t, c in Counter(titles).most_common():
    print(f"  {t}: {c}")

# --- Section B: Average feature values ---
FEATS = ['feat_builder_score', 'feat_retrieval_depth', 'feat_search_relevance_evidence', 'feat_product_exposure']

print(f"\n{'='*70}")
print("B) Average Feature Values")
print(f"{'='*70}")
for f in FEATS:
    vals = elite41[f].to_numpy().astype(float)
    print(f"  {f:<40} mean={np.mean(vals):.3f}  median={np.median(vals):.3f}  std={np.std(vals):.3f}")
yoe = elite41['years_of_experience'].to_numpy().astype(float)
print(f"  {'years_of_experience':<40} mean={np.mean(yoe):.1f}  median={np.median(yoe):.1f}  std={np.std(yoe):.1f}")

# --- Section C: Compare against tiers ---
top20 = df_ranked.filter(pl.col("rank") < 20)
top100 = df_ranked.filter(pl.col("rank") < 100)
rest = df_ranked.filter(pl.col("rank") >= 20)

groups = {
    "Elite 41": elite41,
    "Top 20": top20,
    "Top 100": top100,
    "Full Pool": df_ranked,
}

print(f"\n{'='*100}")
print("C) Comparison Table")
print(f"{'='*100}")
print(f"{'Group':<15} {'n':>6} {'Builder':>10} {'RetD':>10} {'SR':>10} {'ProdExp':>10} {'YoE':>8}")
print("-"*70)
for name, grp in groups.items():
    b = np.mean(grp['feat_builder_score'].to_numpy().astype(float))
    r = np.mean(grp['feat_retrieval_depth'].to_numpy().astype(float))
    s = np.mean(grp['feat_search_relevance_evidence'].to_numpy().astype(float))
    p = np.mean(grp['feat_product_exposure'].to_numpy().astype(float))
    y = np.mean(grp['years_of_experience'].to_numpy().astype(float))
    print(f"{name:<15} {len(grp):6} {b:10.3f} {r:10.3f} {s:10.3f} {p:10.3f} {y:8.1f}")

# --- Cohen's d: Elite 41 vs Rest of Pool ---
print(f"\n{'='*70}")
print("Cohen's d: Elite 41 vs Rest of Pool (rank >= 20)")
print(f"{'='*70}")

def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*np.std(a,ddof=1)**2 + (nb-1)*np.std(b,ddof=1)**2) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0

for f in FEATS:
    e_vals = elite41[f].to_numpy().astype(float)
    r_vals = rest[f].to_numpy().astype(float)
    d = cohens_d(e_vals, r_vals)
    print(f"  {f:<40} d = {d:.4f}")

# --- Binary feature evaluation ---
print(f"\n{'='*70}")
print("Binary Feature: elite_ranking_specialist")
print(f"{'='*70}")

# How many of the Top 20 are elite?
top20_elite = top20.filter(pl.col("elite_ranking_specialist") == 1)
top20_non = top20.filter(pl.col("elite_ranking_specialist") == 0)

# How many elite are NOT in the top 20?
elite_in_top20 = len(top20_elite)
elite_not_in_top20 = len(elite41) - elite_in_top20
non_elite_in_top20 = len(top20_non)

print(f"  Elite in Top 20:      {elite_in_top20} / 20")
print(f"  Elite NOT in Top 20:  {elite_not_in_top20} / {len(elite41)}")
print(f"  Non-elite in Top 20:  {non_elite_in_top20} / 20")

# Precision = elite_in_top20 / total_elite
precision = elite_in_top20 / len(elite41) if len(elite41) > 0 else 0
# False positive rate = elite NOT in top20 / total elite
fpr = elite_not_in_top20 / len(elite41) if len(elite41) > 0 else 0
# Recall = elite_in_top20 / top20
recall = elite_in_top20 / 20

print(f"\n  Precision (elite -> Top20):   {precision:.3f} ({precision*100:.1f}%)")
print(f"  Recall (Top20 captured):     {recall:.3f} ({recall*100:.1f}%)")
print(f"  False Positive Rate:         {fpr:.3f} ({fpr*100:.1f}%)")

# Show which Top 20 candidates are elite vs not
print(f"\n  Top 20 Breakdown:")
for row in top20.sort("rank").iter_rows(named=True):
    tag = "[ELITE]" if row['elite_ranking_specialist'] == 1 else "  ---"
    print(f"    Rank {row['rank']+1:>2} | {row['candidate_id']} | {str(row.get('current_title','')):<30} | Builder={row['feat_builder_score']:.3f} | CRD={row['clean_ranking_depth']:.3f} | {tag}")

# Cohen's d of binary feature against Top20 vs Rest
top20_binary = top20['elite_ranking_specialist'].to_numpy().astype(float)
rest_binary = rest['elite_ranking_specialist'].to_numpy().astype(float)
d_binary = cohens_d(top20_binary, rest_binary)
corr, pval = stats.pointbiserialr(
    np.concatenate([np.ones(len(top20_binary)), np.zeros(len(rest_binary))]),
    np.concatenate([top20_binary, rest_binary])
)

print(f"\n  Cohen's d (Top20 vs Rest):   {d_binary:.4f}")
print(f"  Point-biserial r:            {corr:.4f} (p={pval:.2e})")

# --- Recommended bonus ---
# The feature should boost but not dominate. If avg Final_Score in top20 ~0.65,
# a 3-5% bonus for elite candidates is reasonable.
top20_scores = top20['Final_Score'].to_numpy().astype(float)
avg_score = np.mean(top20_scores)
print(f"\n  Avg Final Score (Top 20):    {avg_score:.4f}")
print(f"\n  Recommended Score Bonus:")
print(f"    Conservative: multiply by 1.03 (+3%)")
print(f"    Moderate:     multiply by 1.05 (+5%)")
print(f"    Aggressive:   multiply by 1.08 (+8%)")

# Also check: what if we look at top 10, top 5?
for k in [5, 10, 20, 50]:
    topk = df_ranked.filter(pl.col("rank") < k)
    ek = len(topk.filter(pl.col("elite_ranking_specialist") == 1))
    print(f"    Elite in Top {k:>2}: {ek}/{k} ({ek/k*100:.0f}%)")

print("\nDone.")
