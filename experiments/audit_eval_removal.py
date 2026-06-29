import polars as pl
import numpy as np
import faiss
import math
import json
from pathlib import Path

base_dir = Path(r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS")
artifacts_dir = base_dir / 'artifacts'

df_feat = pl.read_parquet(artifacts_dir / 'features.parquet', glob=False)
df_parsed = pl.read_parquet(artifacts_dir / 'parsed_candidates.parquet', glob=False)

df_pool = df_feat.join(df_parsed, on="candidate_id", how="inner")

from sentence_transformers import SentenceTransformer
jd_text = "Senior AI Engineer Search Ranking Retrieval Embeddings NDCG Vector Databases"
model = SentenceTransformer('all-MiniLM-L6-v2')
jd_emb = model.encode([jd_text])
faiss.normalize_L2(jd_emb)

emb_recent = np.load(artifacts_dir / 'embeddings_recent.npy')
emb_last_two = np.load(artifacts_dir / 'embeddings_last_two.npy')
emb_full = np.load(artifacts_dir / 'embeddings_full.npy')

faiss.normalize_L2(emb_recent)
faiss.normalize_L2(emb_last_two)
faiss.normalize_L2(emb_full)

df_pool = df_pool.with_row_index("faiss_id")
pool_ids = df_pool['faiss_id'].to_list()

sim_recent = (emb_recent[pool_ids] @ jd_emb[0]).tolist()
sim_last_two = (emb_last_two[pool_ids] @ jd_emb[0]).tolist()
sim_full = (emb_full[pool_ids] @ jd_emb[0]).tolist()

df_pool = df_pool.with_columns([
    pl.Series("sim_recent", sim_recent),
    pl.Series("sim_last_two", sim_last_two),
    pl.Series("sim_full", sim_full),
])

df_pool = df_pool.with_columns(
    Base_Score = (0.55 * pl.col("sim_recent")) + (0.30 * pl.col("sim_last_two")) + (0.15 * pl.col("sim_full"))
)

def evaluate_pipeline(df, with_eval=True):
    df = df.with_columns(
        feat_search_builder = ((0.4 * pl.col("feat_builder_score") + 0.3 * pl.col("feat_ranking_depth") + 0.3 * pl.col("feat_retrieval_depth")).clip(0.0, 1.0))
    )
    
    if with_eval:
        df = df.with_columns(
            Technical_Multiplier = 1.0 + 0.25 * pl.col("feat_search_relevance_evidence") + 0.20 * pl.col("feat_ranking_depth") + 0.15 * pl.col("feat_retrieval_depth") + 0.15 * pl.col("feat_evaluation_rigor") + 0.35 * pl.col("feat_builder_score")
        )
    else:
        df = df.with_columns(
            Technical_Multiplier = 1.0 + 0.25 * pl.col("feat_search_relevance_evidence") + 0.20 * pl.col("feat_ranking_depth") + 0.15 * pl.col("feat_retrieval_depth") + 0.35 * pl.col("feat_builder_score")
        )
        
    df = df.with_columns(
        Trajectory_Multiplier = 0.50 + 0.20 * pl.col("feat_builder_score") + 0.20 * pl.col("feat_ranking_depth") + 0.10 * pl.col("feat_retrieval_depth"),
        exempt_contra = pl.when(
            (pl.col("feat_search_builder") >= 0.80) | 
            ((pl.col("feat_ranking_depth") >= 0.80) & (pl.col("feat_retrieval_depth") >= 0.80))
        ).then(pl.col("contradiction_score") * 0.25).otherwise(pl.col("contradiction_score"))
    )
    
    df = df.with_columns(
        Behavioral_Multiplier = 1.0 + (pl.col("feat_availability_score") + pl.col("feat_saved_boost") + pl.col("feat_search_appearance_boost") - 1.0) * 0.25,
        Persona_Penalty = pl.col("feat_wrapper_ai_only") * pl.col("feat_architect_no_coding"),
        Honeypot_Decay = pl.col("exempt_contra").map_elements(lambda x: math.exp(-0.10 * x), return_dtype=pl.Float64)
    )
    
    df = df.with_columns(
        Core_Mult = pl.col("Technical_Multiplier") * pl.col("feat_verified_search_skill") * pl.col("Trajectory_Multiplier") * pl.col("Behavioral_Multiplier") * pl.col("Persona_Penalty") * pl.col("Honeypot_Decay")
    )
    
    df = df.with_columns(Final_Score = pl.col("Core_Mult") * (1.0 + 0.20 * pl.col("Base_Score")))
    return df.sort(["Final_Score", "candidate_id"], descending=[True, False]).with_row_index("rank")

df_base = evaluate_pipeline(df_pool, True)
df_no_eval = evaluate_pipeline(df_pool, False)

# Top 20 comparison
old_20 = df_base.head(20)['candidate_id'].to_list()
new_20 = df_no_eval.head(20)['candidate_id'].to_list()

entrants = set(new_20) - set(old_20)
exits = set(old_20) - set(new_20)

print("A) Top20 comparison")
print(f"Entrants: {len(entrants)}")
for e in entrants:
    r_new = df_no_eval.filter(pl.col("candidate_id")==e)['rank'][0] + 1
    r_old = df_base.filter(pl.col("candidate_id")==e)['rank'][0] + 1
    print(f"  {e}: Rank {r_old} -> {r_new} ({df_base.filter(pl.col('candidate_id')==e)['current_title'][0]})")

print(f"Exits: {len(exits)}")
for e in exits:
    r_new = df_no_eval.filter(pl.col("candidate_id")==e)['rank'][0] + 1
    r_old = df_base.filter(pl.col("candidate_id")==e)['rank'][0] + 1
    print(f"  {e}: Rank {r_old} -> {r_new} ({df_base.filter(pl.col('candidate_id')==e)['current_title'][0]})")

print("Rank Movements (Top 20 survivors):")
for e in set(new_20).intersection(old_20):
    r_new = df_no_eval.filter(pl.col("candidate_id")==e)['rank'][0] + 1
    r_old = df_base.filter(pl.col("candidate_id")==e)['rank'][0] + 1
    if r_new != r_old:
        print(f"  {e}: Rank {r_old} -> {r_new} ({r_old - r_new:+})")

# Top 100 comparison
def get_personas(df_100):
    titles = df_100['current_title'].fill_null("").to_list()
    search = sum(1 for t in titles if any(w in t.lower() for w in ['search', 'ranking', 'relevance', 'ir engineer', 'information retrieval']))
    recsys = sum(1 for t in titles if any(w in t.lower() for w in ['recommendation', 'recsys']))
    nlp = sum(1 for t in titles if any(w in t.lower() for w in ['nlp', 'natural language']))
    applied_ml = sum(1 for t in titles if 'applied' in t.lower() and ('scientist' in t.lower() or 'ml' in t.lower()))
    junior = sum(1 for t in titles if 'junior' in t.lower() or 'associate' in t.lower())
    cv = sum(1 for t in titles if 'computer vision' in t.lower() or 'vision' in t.lower())
    return search, recsys, nlp, applied_ml, junior, cv

print("\nB) Top100 comparison (Without Evaluation)")
s, r, n, a, j, c = get_personas(df_no_eval.head(100))
o_s, o_r, o_n, o_a, o_j, o_c = get_personas(df_base.head(100))

print(f"Search Engineers: {s} (was {o_s})")
print(f"Recommendation Engineers: {r} (was {o_r})")
print(f"NLP Engineers: {n} (was {o_n})")
print(f"Applied ML Engineers: {a} (was {o_a})")
print(f"Junior contamination: {j} (was {o_j})")
print(f"CV contamination: {c} (was {o_c})")

# Elite concentration
print("\nC) Elite concentration (Without Evaluation)")
def count_elites(df_k):
    return df_k.filter(pl.col("feat_builder_score") >= 0.8).height

e20 = count_elites(df_no_eval.head(20))
e50 = count_elites(df_no_eval.head(50))
e100 = count_elites(df_no_eval.head(100))

o20 = count_elites(df_base.head(20))
o50 = count_elites(df_base.head(50))
o100 = count_elites(df_base.head(100))

print(f"Top 20 Elites: {e20} (was {o20})")
print(f"Top 50 Elites: {e50} (was {o50})")
print(f"Top 100 Elites: {e100} (was {o100})")
