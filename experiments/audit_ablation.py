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

# Recreate Base_Score since it's not saved in features.parquet
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

def get_titles(df):
    return df['current_title'].fill_null("").to_list()

def get_metrics(df_sorted, k):
    df_top = df_sorted.head(k)
    b_mean = df_top['feat_builder_score'].mean()
    r_mean = df_top['feat_retrieval_depth'].mean()
    rd_mean = df_top['feat_ranking_depth'].mean()
    
    titles = get_titles(df_top)
    search = sum(1 for t in titles if any(w in t.lower() for w in ['search', 'ranking', 'relevance', 'ir engineer', 'information retrieval']))
    recsys = sum(1 for t in titles if any(w in t.lower() for w in ['recommendation', 'recsys']))
    nlp = sum(1 for t in titles if any(w in t.lower() for w in ['nlp', 'natural language']))
    
    elites = df_top.filter(pl.col("feat_builder_score") >= 0.8).height
    return b_mean, r_mean, rd_mean, search + recsys + nlp, elites

def evaluate_pipeline(name, df, ablation=None):
    # Compute pieces
    df = df.with_columns(
        feat_search_builder = ((0.4 * pl.col("feat_builder_score") + 0.3 * pl.col("feat_ranking_depth") + 0.3 * pl.col("feat_retrieval_depth")).clip(0.0, 1.0))
    )
    
    if ablation == "no_retrieval":
        df = df.with_columns(
            Technical_Multiplier = 1.0 + 0.25 * pl.col("feat_search_relevance_evidence") + 0.20 * pl.col("feat_ranking_depth") + 0.15 * pl.col("feat_evaluation_rigor") + 0.35 * pl.col("feat_builder_score"),
            Trajectory_Multiplier = 0.50 + 0.20 * pl.col("feat_builder_score") + 0.20 * pl.col("feat_ranking_depth"),
            feat_search_builder = ((0.4 * pl.col("feat_builder_score") + 0.3 * pl.col("feat_ranking_depth")).clip(0.0, 1.0))
        )
    elif ablation == "no_ranking":
        df = df.with_columns(
            Technical_Multiplier = 1.0 + 0.25 * pl.col("feat_search_relevance_evidence") + 0.15 * pl.col("feat_retrieval_depth") + 0.15 * pl.col("feat_evaluation_rigor") + 0.35 * pl.col("feat_builder_score"),
            Trajectory_Multiplier = 0.50 + 0.20 * pl.col("feat_builder_score") + 0.10 * pl.col("feat_retrieval_depth"),
            feat_search_builder = ((0.4 * pl.col("feat_builder_score") + 0.3 * pl.col("feat_retrieval_depth")).clip(0.0, 1.0))
        )
    elif ablation == "no_eval":
        df = df.with_columns(
            Technical_Multiplier = 1.0 + 0.25 * pl.col("feat_search_relevance_evidence") + 0.20 * pl.col("feat_ranking_depth") + 0.15 * pl.col("feat_retrieval_depth") + 0.35 * pl.col("feat_builder_score"),
            Trajectory_Multiplier = 0.50 + 0.20 * pl.col("feat_builder_score") + 0.20 * pl.col("feat_ranking_depth") + 0.10 * pl.col("feat_retrieval_depth")
        )
    elif ablation == "no_traj":
        df = df.with_columns(
            Technical_Multiplier = 1.0 + 0.25 * pl.col("feat_search_relevance_evidence") + 0.20 * pl.col("feat_ranking_depth") + 0.15 * pl.col("feat_retrieval_depth") + 0.15 * pl.col("feat_evaluation_rigor") + 0.35 * pl.col("feat_builder_score"),
            Trajectory_Multiplier = pl.lit(1.0)
        )
    else:
        # standard
        df = df.with_columns(
            Technical_Multiplier = 1.0 + 0.25 * pl.col("feat_search_relevance_evidence") + 0.20 * pl.col("feat_ranking_depth") + 0.15 * pl.col("feat_retrieval_depth") + 0.15 * pl.col("feat_evaluation_rigor") + 0.35 * pl.col("feat_builder_score"),
            Trajectory_Multiplier = 0.50 + 0.20 * pl.col("feat_builder_score") + 0.20 * pl.col("feat_ranking_depth") + 0.10 * pl.col("feat_retrieval_depth")
        )
        
    if ablation == "no_elite_exemption":
        df = df.with_columns(exempt_contra = pl.col("contradiction_score"))
    else:
        df = df.with_columns(
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
    
    if ablation == "no_variant_d":
        df = df.with_columns(Final_Score = pl.col("Core_Mult") * pl.col("Base_Score"))
    else:
        df = df.with_columns(Final_Score = pl.col("Core_Mult") * (1.0 + 0.20 * pl.col("Base_Score")))
        
    df_sorted = df.sort(["Final_Score", "candidate_id"], descending=[True, False])
    
    metrics_20 = get_metrics(df_sorted, 20)
    metrics_50 = get_metrics(df_sorted, 50)
    metrics_100 = get_metrics(df_sorted, 100)
    
    return metrics_20, metrics_50, metrics_100

ablations = [
    ("Baseline", None),
    ("No Trajectory", "no_traj"),
    ("No Elite Exemption", "no_elite_exemption"),
    ("No Variant D", "no_variant_d"),
    ("No Retrieval", "no_retrieval"),
    ("No Ranking", "no_ranking"),
    ("No Evaluation", "no_eval")
]

print("Ablation Study Results\n")
print(f"{'Scenario':<25} | {'Top 20 Elites':<15} | {'Top 100 Elites':<15} | {'Top 100 Target Personas':<25} | {'Top 100 Avg Rank/Ret':<20}")
print("-" * 110)

base_m20, base_m50, base_m100 = evaluate_pipeline("Baseline", df_pool, None)

for name, ab in ablations:
    m20, m50, m100 = evaluate_pipeline(name, df_pool, ab)
    
    elites_20 = m20[4]
    elites_100 = m100[4]
    target_100 = m100[3] # search+recsys+nlp
    avg_rank_ret = (m100[2] + m100[1]) / 2.0
    
    # Delta vs base
    if name != "Baseline":
        d_elites_20 = elites_20 - base_m20[4]
        d_elites_100 = elites_100 - base_m100[4]
        d_target_100 = target_100 - base_m100[3]
        d_arr = avg_rank_ret - ((base_m100[2] + base_m100[1]) / 2.0)
        print(f"{name:<25} | {elites_20:<5} ({d_elites_20:>+3})      | {elites_100:<5} ({d_elites_100:>+3})      | {target_100:<5} ({d_target_100:>+3})                 | {avg_rank_ret:.3f} ({d_arr:>+.3f})")
    else:
        print(f"{name:<25} | {elites_20:<15} | {elites_100:<15} | {target_100:<25} | {avg_rank_ret:.3f}")
