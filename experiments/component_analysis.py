import polars as pl
from pathlib import Path
import numpy as np

artifacts_dir = Path(r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS\artifacts")
df_debug = pl.read_parquet(artifacts_dir / 'debug_top200.parquet', glob=False)
df_top50 = df_debug.head(50)

components = []

for row in df_top50.iter_rows(named=True):
    base_score = row['Base_Score']
    
    # Technical Multiplier Additive Components
    builder_contrib = 0.35 * row['feat_builder_score']
    search_rel_contrib = 0.25 * row['feat_search_relevance_evidence']
    rank_depth_contrib = 0.20 * row['feat_ranking_depth']
    ret_depth_contrib = 0.15 * row['feat_retrieval_depth']
    eval_rigor_contrib = 0.15 * row['feat_evaluation_rigor']
    
    tech_mult = 1.0 + builder_contrib + search_rel_contrib + rank_depth_contrib + ret_depth_contrib + eval_rigor_contrib
    
    # Multipliers
    behavioral_mult = row['feat_availability_score'] + row['feat_saved_boost'] + row['feat_search_appearance_boost']
    trajectory_mult = row['feat_product_exposure'] + row['feat_trajectory_transition']
    verified_search = row['feat_verified_search_skill']
    
    # To compare dominance, we can look at the relative size of multipliers compared to 1.0, 
    # and additive components compared to the total tech multiplier.
    
    c = {
        'candidate_id': row['candidate_id'],
        'title': row['current_title'],
        'base_score': base_score,
        'builder_contrib': builder_contrib,
        'search_rel_contrib': search_rel_contrib,
        'rank_depth_contrib': rank_depth_contrib,
        'ret_depth_contrib': ret_depth_contrib,
        'eval_rigor_contrib': eval_rigor_contrib,
        'behavioral_mult': behavioral_mult,
        'trajectory_mult': trajectory_mult,
        'verified_search': verified_search,
        'final_score': row['Final_Score']
    }
    components.append(c)

# Calculate averages for features to find strongest/weakest contributors
avgs = {
    'Semantic (Base_Score)': np.mean([c['base_score'] for c in components]),
    'Builder Contrib': np.mean([c['builder_contrib'] for c in components]),
    'Search Rel Contrib': np.mean([c['search_rel_contrib'] for c in components]),
    'Rank Depth Contrib': np.mean([c['rank_depth_contrib'] for c in components]),
    'Ret Depth Contrib': np.mean([c['ret_depth_contrib'] for c in components]),
    'Eval Rigor Contrib': np.mean([c['eval_rigor_contrib'] for c in components]),
    'Behavioral Mult': np.mean([c['behavioral_mult'] for c in components]),
    'Trajectory Mult': np.mean([c['trajectory_mult'] for c in components]),
    'Verified Search Mult': np.mean([c['verified_search'] for c in components])
}

# Sort averages to find strongest/weakest
sorted_avgs = sorted(avgs.items(), key=lambda x: x[1], reverse=True)

print("--- 1. Top Strongest Contributors (Average Value) ---")
for k, v in sorted_avgs:
    print(f"{k}: {v:.4f}")

print("\n--- 2. Top Weakest Contributors (Average Value) ---")
for k, v in reversed(sorted_avgs):
    print(f"{k}: {v:.4f}")

# Domination metrics
# Semantic domination: Base score is very high (> 0.2) but multipliers/tech are low
# Builder domination: builder_contrib is the largest additive component in tech mult
# Behavioral domination: behavioral_mult is exceptionally high (> 1.5)

print("\n--- 3. Candidates dominated by Semantic Similarity ---")
sem_dom = sorted(components, key=lambda c: c['base_score'] / c['final_score'] if c['final_score']>0 else 0, reverse=True)[:5]
for c in sem_dom:
    print(f"{c['candidate_id']} ({c['title']}): Base Score = {c['base_score']:.3f}, Final Score = {c['final_score']:.3f}")

print("\n--- 4. Candidates dominated by Builder Signals ---")
# Where builder contrib is the largest component of tech_mult (excluding the base 1.0)
builder_dom = []
for c in components:
    tech_parts = {'builder': c['builder_contrib'], 'search': c['search_rel_contrib'], 'rank': c['rank_depth_contrib'], 'ret': c['ret_depth_contrib'], 'eval': c['eval_rigor_contrib']}
    max_tech = max(tech_parts.values())
    if max_tech == c['builder_contrib'] and max_tech > 0.1:
        builder_dom.append(c)
builder_dom = sorted(builder_dom, key=lambda c: c['builder_contrib'], reverse=True)[:5]
for c in builder_dom:
    print(f"{c['candidate_id']} ({c['title']}): Builder Contrib = {c['builder_contrib']:.3f}, Next Best Contrib = {sorted([c['search_rel_contrib'], c['rank_depth_contrib'], c['ret_depth_contrib'], c['eval_rigor_contrib']])[-1]:.3f}")

print("\n--- 5. Candidates dominated by Behavioral Signals ---")
behav_dom = sorted(components, key=lambda c: c['behavioral_mult'], reverse=True)[:5]
for c in behav_dom:
    print(f"{c['candidate_id']} ({c['title']}): Behavioral Mult = {c['behavioral_mult']:.3f}, Final Score = {c['final_score']:.3f}")

