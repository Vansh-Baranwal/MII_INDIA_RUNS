import polars as pl
import numpy as np
import faiss
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from collections import defaultdict, Counter

artifacts_dir = Path(r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS\artifacts")

# Read pool and features
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
df_pool = df_pool.join(df_parsed, on="candidate_id", how="left")

# Base calculations for Ranking
df_pool = df_pool.with_columns([
    (0.55*pl.col("sim_recent") + 0.30*pl.col("sim_last_two") + 0.15*pl.col("sim_full")).alias("Base_Score"),
    ((0.4 * pl.col("feat_builder_score") + 0.3 * pl.col("feat_ranking_depth") + 0.3 * pl.col("feat_retrieval_depth")).clip(0.0, 1.0)).alias("feat_search_builder"),
    (1.0 + (pl.col("feat_availability_score") + pl.col("feat_saved_boost") + pl.col("feat_search_appearance_boost") - 1.0)*0.25).alias("Behavioral_Multiplier"),
    (pl.col("feat_wrapper_ai_only") * pl.col("feat_architect_no_coding")).alias("Persona_Penalty")
])

df_pool = df_pool.with_columns([
    (0.50 + 0.50 * pl.col("feat_search_builder")).alias("Trajectory_Multiplier"),
    (1.0 + (0.25*pl.col("feat_search_relevance_evidence") + 0.20*pl.col("feat_ranking_depth") + 0.15*pl.col("feat_retrieval_depth") + 0.15*pl.col("feat_evaluation_rigor") + 0.35*pl.col("feat_builder_score"))).alias("Technical_Multiplier")
])

def evaluate_contradiction_logic(row, exclude_rule=None):
    score = 0.0
    yoe = row.get('years_of_experience_right', row.get('years_of_experience', 0))
    if yoe is None: yoe = 0
    dur_total = row.get('total_duration_months', 0)
    if dur_total is None: dur_total = 0
    grad_year = row.get('grad_year', 0)
    if grad_year is None: grad_year = 0
    
    triggered = []
    
    if dur_total > (yoe * 12) * 1.2:
        if exclude_rule != 'dur_total':
            score += 5.0
        triggered.append('dur_total')
            
    if grad_year > 0:
        if (2026 - grad_year) < yoe - 2:
            if exclude_rule != 'grad_year':
                score += 5.0
            triggered.append('grad_year')
            
    sal_min = row.get('expected_salary_min', 0)
    if sal_min is None: sal_min = 0
    if yoe <= 2 and sal_min > 40:
        if exclude_rule != 'sal_min':
            score += 5.0
        triggered.append('sal_min')
        
    views = row.get('profile_views_received_30d', 0)
    if views is None: views = 0
    response_rate = row.get('recruiter_response_rate', 0.0)
    if response_rate is None: response_rate = 0.0
    if response_rate == 1.0 and views == 0:
        if exclude_rule != 'ghost_profile':
            score += 2.0
        triggered.append('ghost_profile')
        
    try:
        skills = json.loads(row.get('skills_json', '[]'))
        for s in skills:
            if s.get('proficiency') == 'expert' and s.get('duration_months', 0) == 0:
                if exclude_rule != 'fake_expert':
                    score += 1.0
                triggered.append('fake_expert')
    except:
        pass

    return score, triggered

# Calculate base ranks (Baseline)
baseline_scores = []
rule_triggers_map = defaultdict(list)

for row in df_pool.iter_rows(named=True):
    cid = row['candidate_id']
    score, triggered = evaluate_contradiction_logic(row)
    baseline_scores.append(score)
    for t in triggered:
        rule_triggers_map[t].append(cid)

df_pool = df_pool.with_columns([
    pl.Series("calc_contradiction_score", baseline_scores)
])

def rank_pool(df, contra_col):
    df_sim = df.with_columns([
        ((-0.10 * pl.col(contra_col)).exp()).alias("Honeypot_Decay")
    ])
    df_sim = df_sim.with_columns(
        (pl.col("Base_Score") * pl.col("Technical_Multiplier") * pl.col("feat_verified_search_skill") * pl.col("Trajectory_Multiplier") * pl.col("Behavioral_Multiplier") * pl.col("Persona_Penalty") * pl.col("Honeypot_Decay")).alias("Final_Score")
    )
    df_ranked = df_sim.sort(["Final_Score","candidate_id"], descending=[True,False]).with_row_index("rank")
    return df_ranked

df_baseline = rank_pool(df_pool, "calc_contradiction_score")

print("\n" + "="*80)
print("CONTRADICTION RULE TRIGGER ANALYSIS")
print("="*80)

for rule, cids in rule_triggers_map.items():
    rule_df = df_baseline.filter(pl.col("candidate_id").is_in(cids))
    if len(rule_df) == 0: continue
    
    avg_rank = rule_df['rank'].mean() + 1
    avg_b = rule_df['feat_builder_score'].mean()
    avg_r = rule_df['feat_retrieval_depth'].mean()
    avg_rd = rule_df['feat_ranking_depth'].mean()
    
    print(f"\nRule: {rule}")
    print(f"  Count: {len(cids)} candidates")
    print(f"  Avg Rank: {avg_rank:.1f}")
    print(f"  Avg Builder Score: {avg_b:.3f}")
    print(f"  Avg Retrieval Depth: {avg_r:.3f}")
    print(f"  Avg Ranking Depth: {avg_rd:.3f}")

print("\n" + "="*80)
print("ELITE CANDIDATES PENALIZED BY RULES (Top 50 by Quality)")
print("="*80)
elite_penalized = df_baseline.filter(
    (pl.col("calc_contradiction_score") >= 3) & 
    ((pl.col("feat_builder_score") >= 0.7) | (pl.col("feat_search_builder") >= 0.8))
).sort("feat_search_builder", descending=True).head(50)

for row in elite_penalized.iter_rows(named=True):
    _, triggered = evaluate_contradiction_logic(row)
    print(f"  {row['candidate_id']} | {row['current_title']:<35} | Rank: {row['rank']+1:>4} | Triggers: {triggered}")

print("\n" + "="*80)
print("SIMULATIONS: REMOVE SINGLE RULE")
print("="*80)

rules_to_test = ['grad_year', 'dur_total', 'sal_min']

for rule in rules_to_test:
    sim_scores = []
    for row in df_pool.iter_rows(named=True):
        score, _ = evaluate_contradiction_logic(row, exclude_rule=rule)
        sim_scores.append(score)
        
    df_sim_pool = df_pool.with_columns(pl.Series("sim_contra", sim_scores))
    df_sim_ranked = rank_pool(df_sim_pool, "sim_contra")
    top100 = df_sim_ranked.head(100)
    
    b_mean = top100['feat_builder_score'].mean()
    r_mean = top100['feat_retrieval_depth'].mean()
    rd_mean = top100['feat_ranking_depth'].mean()
    yoe_mean = top100['years_of_experience_right'].mean()
    
    titles100 = top100['current_title'].fill_null("").to_list()
    search = sum(1 for t in titles100 if any(k in t.lower() for k in ['search', 'ranking', 'relevance', 'ir engineer', 'information retrieval']))
    recsys = sum(1 for t in titles100 if any(k in t.lower() for k in ['recommendation', 'recsys']))
    nlp = sum(1 for t in titles100 if any(k in t.lower() for k in ['nlp', 'natural language']))
    aml = sum(1 for t in titles100 if any(k in t.lower() for k in ['applied ml', 'applied machine learning', 'applied scientist']))
    cv = sum(1 for t in titles100 if any(k in t.lower() for k in ['computer vision', 'vision']))
    junior = sum(1 for t in titles100 if 'junior' in t.lower() or 'associate' in t.lower())
    
    print(f"\nScenario: Remove '{rule}'")
    print(f"  Avg Builder:   {b_mean:.3f}")
    print(f"  Avg Retrieval: {r_mean:.3f}")
    print(f"  Avg Ranking:   {rd_mean:.3f}")
    print(f"  Avg YoE:       {yoe_mean:.1f}")
    print(f"  Top 100 Personas: Search({search}), Rec({recsys}), NLP({nlp}), AML({aml}), Junior({junior}), CV({cv})")

print("\nDone.")
