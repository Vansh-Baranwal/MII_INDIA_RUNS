import polars as pl
import numpy as np
import math
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from collections import Counter

artifacts_dir = Path(r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS\artifacts")

# --- Reconstruct Scenario D Top 100 / Top 20 ---
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
df_ranked = df_pool.sort(["Final_Score","candidate_id"], descending=[True,False])
top100_ids = set(df_ranked.head(100)['candidate_id'].to_list())
top20_ids = set(df_ranked.head(20)['candidate_id'].to_list())

# --- Load full dataset ---
df_all = pl.read_parquet(artifacts_dir / 'parsed_candidates.parquet', glob=False)
total = len(df_all)

# --- Audit terms ---
AUDIT_TERMS = [
    "learning-to-rank", "learning to rank",
    "lambdarank", "lambdamart",
    "ranknet",
    "pairwise ranking", "listwise ranking", "pointwise ranking",
    "click model", "click-through rate prediction", "ctr prediction",
    "position bias",
    "relevance model",
    "query ranking", "search ranking", "ranking system", "ranking pipeline",
    "candidate ranking", "retrieval ranking",
    "reranking", "re-ranking",
    "two-stage retrieval", "multi-stage ranking",
    "recall stage", "precision stage",
    "search relevance",
    "ndcg", "mrr", "dcg", "map@k",
    "ranking evaluation",
]

global_counts = {t: 0 for t in AUDIT_TERMS}
top100_counts = {t: 0 for t in AUDIT_TERMS}
top20_counts = {t: 0 for t in AUDIT_TERMS}

# Also compute new dictionary scores per candidate for the simulation
# Proposed tiers
TIER1 = {"learning-to-rank": 1.0, "learning to rank": 1.0, "lambdarank": 1.0, "lambdamart": 1.0, "ranknet": 1.0}
TIER2 = {"pairwise ranking": 0.9, "listwise ranking": 0.9, "pointwise ranking": 0.9, "click model": 0.9, "ctr prediction": 0.8, "click-through rate prediction": 0.8, "position bias": 0.9, "ndcg": 0.8, "mrr": 0.8, "dcg": 0.8, "ranking evaluation": 0.8}
TIER3 = {"re-ranking": 0.5, "reranking": 0.5, "search ranking": 0.5, "ranking system": 0.5, "ranking pipeline": 0.5, "multi-stage ranking": 0.6, "two-stage retrieval": 0.6, "search relevance": 0.4, "relevance model": 0.5, "recall stage": 0.4, "precision stage": 0.4, "query ranking": 0.5, "candidate ranking": 0.4, "retrieval ranking": 0.4, "map@k": 0.6}

NEW_DICT = {}
NEW_DICT.update(TIER1)
NEW_DICT.update(TIER2)
NEW_DICT.update(TIER3)

# We need to track per-candidate new_ranking_depth for the simulation
# But we also need the features for those candidates (builder, retrieval, title)
# So let's build a map: candidate_id -> (new_raw_score, title, builder, retrieval)

# First pass: join features to all candidates
feat_map = {}
for row in df_feat.iter_rows(named=True):
    feat_map[row['candidate_id']] = {
        'builder': row['feat_builder_score'],
        'retrieval': row['feat_retrieval_depth'],
    }

new_scores = {}  # cid -> new_ranking_raw
print("Scanning 100,000 candidates...")

for row in df_all.iter_rows(named=True):
    cid = row['candidate_id']
    txt = row['full_profile_text'].lower()
    title = row.get('current_title', '')

    for t in AUDIT_TERMS:
        if t in txt:
            global_counts[t] += 1
            if cid in top100_ids:
                top100_counts[t] += 1
            if cid in top20_ids:
                top20_counts[t] += 1

    # Compute new dictionary score
    raw = 0.0
    for term, w in NEW_DICT.items():
        c = txt.count(term)
        raw += c * w
    new_scores[cid] = {'raw': raw, 'norm': min(raw / 3.0, 1.0), 'title': title}

# --- Output Section 1: Term Frequencies ---
print(f"\n{'='*100}")
print(f"  Term Frequency Audit (n={total:,})")
print(f"{'='*100}")
print(f"{'Term':<35} {'Global':>8} {'Top100':>8} {'Top20':>7} {'G%':>7} {'T100%':>7} {'Lift':>7}")
print("-"*80)

for t in AUDIT_TERMS:
    g = global_counts[t]
    t100 = top100_counts[t]
    t20 = top20_counts[t]
    g_pct = g / total * 100
    t100_pct = t100 / 100 * 100
    lift = t100_pct / g_pct if g_pct > 0 else 0
    print(f"{t:<35} {g:8} {t100:8} {t20:7} {g_pct:7.3f} {t100_pct:7.1f} {lift:7.1f}x")

# --- Proposed Dictionary ---
print(f"\n{'='*100}")
print(f"  Proposed Replacement Dictionary")
print(f"{'='*100}")

print("\nTier 1 — Unambiguous Ranking Signals (weight 1.0):")
for t, w in TIER1.items():
    print(f"  '{t}': {w}  (global freq: {global_counts.get(t, 'N/A')})")

print("\nTier 2 — Strong Ranking Signals (weight 0.8-0.9):")
for t, w in TIER2.items():
    print(f"  '{t}': {w}  (global freq: {global_counts.get(t, 'N/A')})")

print("\nTier 3 — Weak Supporting Signals (weight 0.4-0.6):")
for t, w in TIER3.items():
    print(f"  '{t}': {w}  (global freq: {global_counts.get(t, 'N/A')})")

# --- Simulation: who scores >= 0.90 with the new dictionary? ---
# Only consider candidates in the retrieved pool
pool_cids = set(df_pool['candidate_id'].to_list())

high_new = []
for cid in pool_cids:
    if cid in new_scores and new_scores[cid]['norm'] >= 0.90:
        fm = feat_map.get(cid, {})
        high_new.append({
            'cid': cid,
            'title': new_scores[cid]['title'],
            'norm': new_scores[cid]['norm'],
            'raw': new_scores[cid]['raw'],
            'builder': fm.get('builder', 0),
            'retrieval': fm.get('retrieval', 0),
        })

high_new.sort(key=lambda x: x['norm'], reverse=True)

print(f"\n{'='*100}")
print(f"  Simulation: New Dictionary — Candidates with ranking_depth >= 0.90")
print(f"{'='*100}")
print(f"  Count: {len(high_new)}")

if high_new:
    titles = Counter(h['title'] for h in high_new)
    builders = [h['builder'] for h in high_new]
    retrievals = [h['retrieval'] for h in high_new]

    print(f"  Avg Builder Score:   {np.mean(builders):.3f} (median: {np.median(builders):.3f})")
    print(f"  Avg Retrieval Depth: {np.mean(retrievals):.3f} (median: {np.median(retrievals):.3f})")

    print(f"\n  Title Distribution:")
    for t, c in titles.most_common():
        print(f"    {t}: {c}")

    print(f"\n  Top 30 candidates:")
    print(f"  {'ID':<16} {'Title':<35} {'NewRD':>7} {'Raw':>7} {'Builder':>8} {'RetD':>7}")
    for h in high_new[:30]:
        print(f"  {h['cid']:<16} {h['title']:<35} {h['norm']:7.3f} {h['raw']:7.1f} {h['builder']:8.3f} {h['retrieval']:7.3f}")

# Also compute: how many in OLD dictionary scored >= 0.90
old_high = len(df_pool.filter(pl.col("feat_ranking_depth") >= 0.90))
print(f"\n  Comparison:")
print(f"    OLD dictionary: {old_high} candidates >= 0.90")
print(f"    NEW dictionary: {len(high_new)} candidates >= 0.90")
print(f"    Reduction: {old_high - len(high_new)} ({(old_high - len(high_new))/old_high*100:.1f}%)")

print("\nDone.")
