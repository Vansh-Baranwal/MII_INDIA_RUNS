import polars as pl
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from collections import Counter
import json

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

df_parsed = pl.read_parquet(artifacts_dir / 'parsed_candidates.parquet', glob=False)
df_pool = df_pool.join(df_parsed.select(["candidate_id", "current_title", "years_of_experience", "full_profile_text", "total_duration_months", "grad_year", "expected_salary_min", "profile_views_received_30d", "recruiter_response_rate", "skills_json"]), on="candidate_id", how="left")

# Drop right duplicate if exists
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

# Scenario E Multipliers
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

print(f"\n{'='*100}")
print("PART 1 - Contradictions")
print(f"{'='*100}")

c_df = df_ranked.filter((pl.col("contradiction_score") > 0) & (pl.col("rank") >= 500))

# Rules:
# A: dur_total > (yoe * 12) * 1.2
# B: grad_year > 0 and (2026 - grad_year) < yoe - 2
# C: yoe <= 2 and sal_min > 40
# D: response == 1.0 and views == 0
# E: skill expert and dur == 0

c_records = []
for row in c_df.iter_rows(named=True):
    yoe = row.get('years_of_experience', 0)
    dur = row.get('total_duration_months', 0)
    grad = row.get('grad_year', 0)
    sal = row.get('expected_salary_min', 0)
    
    triggers = []
    if dur > (yoe * 12) * 1.2:
        triggers.append(f"dur_total ({dur}) > yoe*12*1.2 ({yoe*12*1.2}) -> +5.0")
    if grad > 0 and (2026 - grad) < yoe - 2:
        triggers.append(f"grad_year ({grad}) too recent for yoe ({yoe}) -> +5.0")
    if yoe <= 2 and sal > 40:
        triggers.append(f"yoe<=2 and sal>40 -> +5.0")
        
    c_records.append({
        'id': row['candidate_id'],
        'title': row['current_title'],
        'rank': row['rank'] + 1,
        'score': row['contradiction_score'],
        'triggers': triggers,
        'dur': dur, 'yoe': yoe, 'grad': grad
    })

for r in c_records[:20]:
    print(f"\n{r['id']} | {r['title']} | Rank: {r['rank']} | Score: {r['score']}")
    for t in r['triggers']:
        class_label = "likely false positive"
        if "grad_year" in t: class_label = "legitimate or edge case (could be part-time/PhD)"
        if "dur_total" in t: class_label = "likely false positive (overlapping jobs/consulting)"
        print(f"  Trigger: {t} [{class_label}]")

print(f"\nTotal buried candidates penalized by contradiction: {len(c_records)}")
print("Many dur_total triggers are likely FALSE POSITIVES because candidates often hold concurrent roles (e.g., advising, open source, consulting) that inflate total duration.")

print(f"\n{'='*100}")
print("PART 2 - Builder Score")
print(f"{'='*100}")

BUILDER_TERMS = ["productionized", "architected", "scaled", "shipped", "designed", "deployed", "built"]
TECH_TERMS = ["aws", "gcp", "azure", "docker", "kubernetes", "pytorch", "tensorflow", "faiss", "milvus", "qdrant", "kafka", "spark"]

top50_b = df_ranked.sort(["feat_builder_score","candidate_id"], descending=[True,False]).head(50)
bot50_b = df_ranked.sort(["feat_builder_score","candidate_id"], descending=[False,True]).head(50)

def analyze_builder_group(group_df, name):
    print(f"\n{name}")
    titles = group_df['current_title'].fill_null("").to_list()
    term_counts = Counter()
    tech_counts = Counter()
    
    for txt in group_df['full_profile_text'].fill_null("").to_list():
        t_low = txt.lower()
        for term in BUILDER_TERMS:
            term_counts[term] += t_low.count(term)
        for tech in TECH_TERMS:
            tech_counts[tech] += 1 if tech in t_low else 0
            
    print(f"  Title Distribution:")
    for t, c in Counter(titles).most_common(5): print(f"    {t}: {c}")
    print(f"  Top Builder Terms:")
    for t, c in term_counts.most_common(5): print(f"    {t}: {c}")
    print(f"  Top Tech Stack:")
    for t, c in tech_counts.most_common(5): print(f"    {t}: {c}")

analyze_builder_group(top50_b, "Top 50 by Builder Score")
analyze_builder_group(bot50_b, "Bottom 50 by Builder Score")

print("\nQuestion: Does builder_score primarily reward A) deployment/platform engineering or B) search/retrieval/ranking engineering?")
print("Answer: A) Deployment/platform engineering. It counts generic verbs like 'built', 'deployed', 'scaled', 'designed'. Candidates working heavily with Docker/AWS/Kubernetes naturally use these terms. Pure algorithmic specialists (ranking, math, research) use them far less, so they receive low builder scores despite being search/ranking experts.")

print(f"\n{'='*100}")
print("PART 3 - Retrieval Recall")
print(f"{'='*100}")

# builder_score >= 0.7 AND clean_ranking_depth >= 0.7 AND rank > 500
missing_df = df_ranked.filter(
    (pl.col("feat_builder_score") >= 0.70) & 
    (pl.col("clean_ranking_depth") >= 0.70) & 
    (pl.col("rank") >= 500)
)

print(f"Found {len(missing_df)} highly qualified candidates buried below rank 500.\n")
print(f"{'ID':<16} {'Title':<35} {'Rank':>6} {'BaseScore':>10} {'FinalScore':>10} {'Reason'}")
print("-" * 100)

for row in missing_df.sort("rank").iter_rows(named=True):
    reason = "Unknown"
    if row['contradiction_score'] > 0: reason = "Contradiction Penalty"
    elif row['Base_Score'] < 0.40: reason = "Extremely Low Semantic Match"
    elif row['Behavioral_Multiplier'] < 0.8: reason = "Behavioral Damping"
    else: reason = "Low Semantic Base Score (multiplicative drag)"
    
    print(f"{row['candidate_id']:<16} {row['current_title']:<35} {row['rank']+1:6} {row['Base_Score']:10.3f} {row['Final_Score']:10.4f} {reason}")

print("\nDone.")
