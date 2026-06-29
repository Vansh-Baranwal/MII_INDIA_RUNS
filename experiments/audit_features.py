import polars as pl
import numpy as np
import math
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from scipy import stats

artifacts_dir = Path(r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS\artifacts")

# --- Reconstruct Scenario D ranked pool ---
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

df = pl.read_parquet(artifacts_dir / 'features.parquet', glob=False).with_row_index("faiss_id")
df_pool = df.filter(pl.col("faiss_id").is_in(list(indices_set)))

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
    Base_Score=(0.55 * pl.col("sim_recent") + 0.30 * pl.col("sim_last_two") + 0.15 * pl.col("sim_full")),
    Behavioral_Multiplier=1.0 + (pl.col("feat_availability_score") + pl.col("feat_saved_boost") + pl.col("feat_search_appearance_boost") - 1.0) * 0.25,
    Trajectory_Multiplier=pl.col("feat_product_exposure") + pl.col("feat_trajectory_transition"),
    Persona_Penalty=pl.col("feat_wrapper_ai_only") * pl.col("feat_architect_no_coding"),
    Honeypot_Decay=(-pl.col("contradiction_score")).exp(),
)
df_pool = df_pool.with_columns(
    Technical_Multiplier=1.0 + (
        0.25 * pl.col("feat_search_relevance_evidence") +
        0.20 * pl.col("feat_ranking_depth") +
        0.15 * pl.col("feat_retrieval_depth") +
        0.15 * pl.col("feat_evaluation_rigor") +
        0.35 * pl.col("feat_builder_score")
    ),
)
df_pool = df_pool.with_columns(
    Final_Score=(
        pl.col("Base_Score") * pl.col("Technical_Multiplier") *
        pl.col("feat_verified_search_skill") * pl.col("Trajectory_Multiplier") *
        pl.col("Behavioral_Multiplier") * pl.col("Persona_Penalty") * pl.col("Honeypot_Decay")
    ),
)

df_ranked = df_pool.sort(["Final_Score", "candidate_id"], descending=[True, False]).with_row_index("rank")

FEATURES = [
    "feat_builder_score", "feat_retrieval_depth", "feat_ranking_depth",
    "feat_search_relevance_evidence", "feat_evaluation_rigor",
    "feat_product_exposure", "feat_trajectory_transition",
    "feat_availability_score", "feat_saved_boost", "feat_search_appearance_boost",
    "contradiction_score", "years_of_experience",
]

# --- 1. Split into tiers ---
top50 = df_ranked.filter(pl.col("rank") < 50)
top100 = df_ranked.filter(pl.col("rank") < 100)
top250 = df_ranked.filter(pl.col("rank") < 250)
top500 = df_ranked.filter(pl.col("rank") < 500)
rest = df_ranked.filter(pl.col("rank") >= 50)

pool_size = len(df_ranked)
print(f"Total retrieved pool: {pool_size}")
print(f"Top 50: {len(top50)} | Top 100: {len(top100)} | Top 250: {len(top250)} | Top 500: {len(top500)} | Rest (>=50): {len(rest)}")

# --- 2. Per-tier statistics ---
def tier_stats(tier_df, tier_name):
    print(f"\n{'='*70}")
    print(f"  {tier_name} (n={len(tier_df)})")
    print(f"{'='*70}")
    print(f"{'Feature':<40} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 80)
    for f in FEATURES:
        col = tier_df[f].to_numpy().astype(float)
        print(f"{f:<40} {np.mean(col):8.4f} {np.median(col):8.4f} {np.std(col):8.4f} {np.min(col):8.4f} {np.max(col):8.4f}")

tier_stats(top50, "Top 50")
tier_stats(top100, "Top 100")
tier_stats(top250, "Top 250")
tier_stats(top500, "Top 500")
tier_stats(rest, "Rest (rank >= 50)")

# --- 3. Effect Size: Top50 vs Rest ---
print(f"\n{'='*70}")
print(f"  Effect Size: Top 50 vs Rest")
print(f"{'='*70}")
print(f"{'Feature':<40} {'Top50 Mean':>10} {'Rest Mean':>10} {'Cohen d':>10} {'RankCorr':>10} {'p-value':>12}")
print("-" * 92)

effect_sizes = {}
for f in FEATURES:
    t50 = top50[f].to_numpy().astype(float)
    r = rest[f].to_numpy().astype(float)

    mean_diff = np.mean(t50) - np.mean(r)
    pooled_std = np.sqrt((np.std(t50, ddof=1)**2 + np.std(r, ddof=1)**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

    # Point-biserial: 1 if top50, 0 if rest
    labels = np.concatenate([np.ones(len(t50)), np.zeros(len(r))])
    values = np.concatenate([t50, r])
    corr, pval = stats.pointbiserialr(labels, values)

    effect_sizes[f] = abs(cohens_d)
    print(f"{f:<40} {np.mean(t50):10.4f} {np.mean(r):10.4f} {cohens_d:10.4f} {corr:10.4f} {pval:12.2e}")

# --- 4. Leaderboard ---
print(f"\n{'='*70}")
print(f"  Feature Predictiveness Leaderboard (|Cohen's d|)")
print(f"{'='*70}")
sorted_feats = sorted(effect_sizes.items(), key=lambda x: x[1], reverse=True)
for i, (f, d) in enumerate(sorted_feats):
    tag = ""
    if d < 0.2:
        tag = " [NOISE]"
    elif d < 0.5:
        tag = " [SMALL]"
    elif d < 0.8:
        tag = " [MEDIUM]"
    else:
        tag = " [LARGE]"
    print(f"  {i+1:>2}. {f:<40} |d| = {d:.4f}{tag}")

# --- 5. Noise features ---
print(f"\n{'='*70}")
print(f"  Noise Features (|Cohen's d| < 0.20)")
print(f"{'='*70}")
for f, d in sorted_feats:
    if d < 0.20:
        t50_mean = np.mean(top50[f].to_numpy().astype(float))
        rest_mean = np.mean(rest[f].to_numpy().astype(float))
        print(f"  {f:<40} Top50={t50_mean:.4f}  Rest={rest_mean:.4f}  |d|={d:.4f}")

# --- 6. Recommendations ---
print(f"\n{'='*70}")
print(f"  Recommendations")
print(f"{'='*70}")
print("\nA. Features to INCREASE weight (|d| >= 0.50, Top50 mean > Rest mean):")
for f, d in sorted_feats:
    t50_mean = np.mean(top50[f].to_numpy().astype(float))
    rest_mean = np.mean(rest[f].to_numpy().astype(float))
    if d >= 0.50 and t50_mean > rest_mean:
        print(f"  - {f} (d={d:.3f}, Top50={t50_mean:.3f} vs Rest={rest_mean:.3f})")

print("\nB. Features to DECREASE weight (Top50 mean < Rest mean OR contradicts ranking intent):")
for f, d in sorted_feats:
    t50_mean = np.mean(top50[f].to_numpy().astype(float))
    rest_mean = np.mean(rest[f].to_numpy().astype(float))
    if t50_mean < rest_mean and d >= 0.20:
        print(f"  - {f} (d={d:.3f}, Top50={t50_mean:.3f} vs Rest={rest_mean:.3f})")

print("\nC. Features to DELETE entirely (|d| < 0.10, pure noise):")
for f, d in sorted_feats:
    if d < 0.10:
        print(f"  - {f} (|d|={d:.4f})")

# --- 7. Estimated NDCG gain from weight increases ---
print(f"\n{'='*70}")
print(f"  Estimated NDCG Impact (Weight Sensitivity)")
print(f"{'='*70}")

# Current weights in Technical_Multiplier
current_weights = {
    "feat_search_relevance_evidence": 0.25,
    "feat_ranking_depth": 0.20,
    "feat_retrieval_depth": 0.15,
    "feat_evaluation_rigor": 0.15,
    "feat_builder_score": 0.35,
}

# For behavioral/trajectory features, approximate their contribution
# For each feature, simulate what happens to the score gap between Top50 and Rest
# when we scale its weight by +25%, +50%, +100%

print(f"\n{'Feature':<40} {'|d| now':>8} {'|d|+25%':>8} {'|d|+50%':>8} {'|d|+100%':>8}")
print("-" * 72)

for f in FEATURES:
    t50 = top50[f].to_numpy().astype(float)
    r = rest[f].to_numpy().astype(float)
    
    base_d = effect_sizes[f]
    
    # Simulate amplification of this feature's contribution
    # A larger weight on a feature with positive Cohen's d increases separation
    # Approximation: scaling weight by X scales the feature's contribution to final score
    # The effect on d is roughly proportional to the weight scaling for that component
    
    t50_mean = np.mean(t50)
    r_mean = np.mean(r)
    direction = 1 if t50_mean >= r_mean else -1
    
    # Simple linear approximation of d scaling
    d_25 = base_d * 1.25 if direction > 0 else base_d * 0.75
    d_50 = base_d * 1.50 if direction > 0 else base_d * 0.50
    d_100 = base_d * 2.00 if direction > 0 else base_d * 0.00
    
    print(f"{f:<40} {base_d:8.4f} {d_25:8.4f} {d_50:8.4f} {d_100:8.4f}")

print("\nDone.")
