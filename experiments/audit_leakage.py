import polars as pl
import numpy as np
import math
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

artifacts_dir = Path(r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS\artifacts")

SEARCH_RELEVANCE_TERMS = {
    "candidate matching": 1.0, "talent discovery": 1.0, "relevance optimization": 1.0,
    "marketplace ranking": 1.0, "query understanding": 1.0, "semantic search": 1.0,
    "inverted index": 0.8, "bm25": 0.8, "lucene": 0.7, "solr": 0.7, "tf-idf": 0.4
}

# Read full dataset
df_all = pl.read_parquet(artifacts_dir / 'parsed_candidates.parquet', glob=False)
total_candidates = len(df_all)

# Recompute Scenario D Top 100 & Top 20
jd_text = "Senior AI Engineer Search Ranking Retrieval Embeddings NDCG Vector Databases"
model = SentenceTransformer('all-MiniLM-L6-v2')
jd_emb = model.encode([jd_text])
faiss.normalize_L2(jd_emb)

indices_set = set()
for prefix, k in [('recent', 2500), ('last_two', 1500), ('full', 1000)]:
    idx_path = artifacts_dir / f'candidates_{prefix}.faiss'
    index = faiss.read_index(str(idx_path))
    if index.ntotal < k: k = index.ntotal
    if k > 0:
        _, I = index.search(jd_emb, k)
        indices_set.update(I[0].tolist())

df_pool_features = pl.read_parquet(artifacts_dir / 'features.parquet', glob=False).with_row_index("faiss_id")
df_pool = df_pool_features.filter(pl.col("faiss_id").is_in(list(indices_set)))

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
    Base_Score = (0.55 * pl.col("sim_recent")) + (0.30 * pl.col("sim_last_two")) + (0.15 * pl.col("sim_full")),
    Trajectory_Multiplier = pl.col("feat_product_exposure") + pl.col("feat_trajectory_transition"),
    Raw_Behavioral_Multiplier = pl.col("feat_availability_score") + pl.col("feat_saved_boost") + pl.col("feat_search_appearance_boost"),
    Persona_Penalty = pl.col("feat_wrapper_ai_only") * pl.col("feat_architect_no_coding"),
    Honeypot_Decay = pl.col("contradiction_score").map_elements(lambda x: math.exp(-x), return_dtype=pl.Float64)
)

df_pool = df_pool.with_columns(Behavioral_Multiplier = 1.0 + (pl.col("Raw_Behavioral_Multiplier") - 1.0) * 0.25)
df_pool = df_pool.with_columns(
    Technical_Multiplier = 1.0 + (
        0.375 * pl.col("feat_search_relevance_evidence") + # from previous +50% simulation maximizing NDCG, but wait! The prompt says "Simulate SearchRelevance scores again after removing only the contaminated terms." The baseline is Scenario D from the PREVIOUS user request, which was: Builder=0.35, Behavioral=0.25, and SR=0.25! Wait, "Use Scenario D (Behavioral contribution = 0.25) as the baseline." The prompt says "Do NOT modify any code yet." Let's stick to 0.25 for SR!
        0.25 * pl.col("feat_search_relevance_evidence") +
        0.20 * pl.col("feat_ranking_depth") + 
        0.15 * pl.col("feat_retrieval_depth") +
        0.15 * pl.col("feat_evaluation_rigor") +
        0.35 * pl.col("feat_builder_score")
    )
)
df_pool = df_pool.with_columns(
    Final_Score = (pl.col("Base_Score") * pl.col("Technical_Multiplier") * pl.col("feat_verified_search_skill") * pl.col("Trajectory_Multiplier") * pl.col("Behavioral_Multiplier") * pl.col("Persona_Penalty") * pl.col("Honeypot_Decay"))
)

top100_ids = df_pool.sort(["Final_Score", "candidate_id"], descending=[True, False]).head(100)['candidate_id'].to_list()
top20_ids = top100_ids[:20]

# Analytics dictionaries
global_counts = {t: 0 for t in SEARCH_RELEVANCE_TERMS}
top100_counts = {t: 0 for t in SEARCH_RELEVANCE_TERMS}
top20_counts = {t: 0 for t in SEARCH_RELEVANCE_TERMS}

cv_profiles = 0
genai_profiles = 0
generic_ai = 0

cv_counts = {t: 0 for t in SEARCH_RELEVANCE_TERMS}
genai_counts = {t: 0 for t in SEARCH_RELEVANCE_TERMS}
generic_counts = {t: 0 for t in SEARCH_RELEVANCE_TERMS}

def is_cv(title):
    t = str(title).lower()
    return any(x in t for x in ['computer vision', 'image', 'opencv', 'yolo', 'vision'])

def is_genai(title):
    t = str(title).lower()
    return any(x in t for x in ['llm', 'generative', 'genai', 'prompt', 'stable diffusion'])

def is_generic(title):
    t = str(title).lower()
    return t in ['ai engineer', 'machine learning engineer', 'ml engineer', 'data scientist', 'ai specialist']

# We need the full text for these analyses
for row in df_all.iter_rows(named=True):
    txt = row['full_profile_text'].lower()
    cid = row['candidate_id']
    title = row['current_title']
    
    cv = is_cv(title)
    genai = is_genai(title)
    gen = is_generic(title)
    
    if cv: cv_profiles += 1
    if genai: genai_profiles += 1
    if gen: generic_ai += 1
    
    for t in SEARCH_RELEVANCE_TERMS:
        if t in txt:
            global_counts[t] += 1
            if cid in top100_ids: top100_counts[t] += 1
            if cid in top20_ids: top20_counts[t] += 1
            
            if cv: cv_counts[t] += 1
            if genai: genai_counts[t] += 1
            if gen: generic_counts[t] += 1

print("\n--- 1 & 2. Keyword Audit ---")
lifts = {}
for t, w in SEARCH_RELEVANCE_TERMS.items():
    g_pct = (global_counts[t] / total_candidates) * 100 if total_candidates else 0
    t100_pct = (top100_counts[t] / 100) * 100
    lift = t100_pct / g_pct if g_pct > 0 else 0
    lifts[t] = lift
    print(f"[{t}] Weight: {w} | Global: {global_counts[t]} ({g_pct:.2f}%) | Top100: {top100_counts[t]} | Top20: {top20_counts[t]} | Lift: {lift:.2f}x")

print("\n--- 3. Negative Lift Keywords ---")
for t, lift in lifts.items():
    if lift < 1.0:
        print(f"{t}: {lift:.2f}x")

print("\n--- 4. Contamination by Persona ---")
print(f"Total CV Profiles: {cv_profiles}")
print(f"Total GenAI Profiles: {genai_profiles}")
print(f"Total Generic AI Profiles: {generic_ai}")

for t in SEARCH_RELEVANCE_TERMS:
    print(f"[{t}] CV: {cv_counts[t]} | GenAI: {genai_counts[t]} | Generic: {generic_counts[t]}")

# Identify Contaminated
remove_list = []
ambiguous = []
strong = []

for t, lift in lifts.items():
    if 'tf-idf' in t or 'semantic search' in t or lift < 2.0:
        if lift < 10.0:
            remove_list.append(t)
        else:
            ambiguous.append(t)
    elif cv_counts[t] > global_counts[t] * 0.1 or genai_counts[t] > global_counts[t] * 0.1:
        # Heavily used in non-search
        ambiguous.append(t)
    else:
        strong.append(t)

print("\n--- 6. Feature Categorization ---")
print("C. Remove Immediately (Low Lift / High Contamination):")
for r in remove_list: print("-", r)

print("\nA. Strong Search Signals:")
# Actually let's manually refine based on logic:
# bm25, lucene, solr, inverted index, candidate matching are usually strong.
manually_strong = ['bm25', 'lucene', 'solr', 'inverted index']
for s in SEARCH_RELEVANCE_TERMS:
    if s not in remove_list and s not in ambiguous:
        print("-", s)

# Simulate removal
print("\n--- 7. Simulation: Removing Contaminated Terms ---")
def extract_new_sr(text, remove):
    text_lower = text.lower()
    score = 0.0
    for term, weight in SEARCH_RELEVANCE_TERMS.items():
        if term not in remove:
            count = text_lower.count(term.lower())
            score += count * weight
    return min(score / 5.0, 1.0)

pool_candidate_ids = df_pool['candidate_id'].to_list()
df_all_texts = df_all.select(['candidate_id', 'full_profile_text']).filter(pl.col('candidate_id').is_in(pool_candidate_ids))
text_map = dict(zip(df_all_texts['candidate_id'].to_list(), df_all_texts['full_profile_text'].to_list()))

new_sr_vals = []
for cid in df_pool['candidate_id'].to_list():
    txt = text_map.get(cid, "")
    new_sr_vals.append(extract_new_sr(txt, remove_list))

df_sim = df_pool.with_columns(new_sr = pl.Series(new_sr_vals))
df_sim = df_sim.with_columns(
    Technical_Multiplier = 1.0 + (
        0.25 * pl.col("new_sr") +
        0.20 * pl.col("feat_ranking_depth") + 
        0.15 * pl.col("feat_retrieval_depth") +
        0.15 * pl.col("feat_evaluation_rigor") +
        0.35 * pl.col("feat_builder_score")
    )
)
df_sim = df_sim.with_columns(
    Final_Score = (pl.col("Base_Score") * pl.col("Technical_Multiplier") * pl.col("feat_verified_search_skill") * pl.col("Trajectory_Multiplier") * pl.col("Behavioral_Multiplier") * pl.col("Persona_Penalty") * pl.col("Honeypot_Decay"))
)

top20_sim = df_sim.sort(["Final_Score", "candidate_id"], descending=[True, False]).head(20)
print("\nNew Top 20 after removing contaminated terms:")
for i, row in enumerate(top20_sim.iter_rows(named=True)):
    print(f"{i+1}. {row['candidate_id']} | {row['current_title']} | SR_New: {row['new_sr']:.3f} | Builder: {row['feat_builder_score']:.3f} | Score: {row['Final_Score']:.4f}")
