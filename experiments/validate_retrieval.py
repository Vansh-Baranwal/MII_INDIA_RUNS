import faiss
import numpy as np
import os
import polars as pl
from pathlib import Path
from sentence_transformers import SentenceTransformer

artifacts_dir = Path(r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS\artifacts")
base_dir = artifacts_dir.parent.parent

# 1. Embed JD
jd_txt_path = base_dir / 'read_docx_output.txt'
if jd_txt_path.exists():
    with open(jd_txt_path, 'r', encoding='utf-8') as f:
        jd_text = f.read()
else:
    jd_text = "Senior AI Engineer Search Ranking Retrieval Embeddings NDCG Vector Databases"

print("Loading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
jd_emb = model.encode([jd_text])
faiss.normalize_L2(jd_emb)

# 2. Query indices
configs = [
    ('candidates_recent.faiss', 1500),
    ('candidates_last_two.faiss', 1000),
    ('candidates_full.faiss', 500)
]

results = {}
retrieved_ids = {}

for idx_name, k in configs:
    idx_path = artifacts_dir / idx_name
    index = faiss.read_index(str(idx_path))
    D, I = index.search(jd_emb, k)
    
    ids = I[0].tolist()
    scores = D[0].tolist()
    
    retrieved_ids[idx_name] = set(ids)
    results[idx_name] = {
        'ids': ids,
        'scores': scores,
        'min_score': min(scores),
        'max_score': max(scores)
    }

# 3. Report overlaps
recent_set = retrieved_ids['candidates_recent.faiss']
last_two_set = retrieved_ids['candidates_last_two.faiss']
full_set = retrieved_ids['candidates_full.faiss']

union_set = recent_set | last_two_set | full_set
union_size = len(union_set)

all_three = recent_set & last_two_set & full_set
recent_and_last_two = recent_set & last_two_set
recent_only = recent_set - (last_two_set | full_set)
full_only = full_set - (recent_set | last_two_set)

# Map FAISS IDs back to Candidate IDs using features.parquet to satisfy "top 20 candidate IDs"
with open(artifacts_dir / 'features.parquet', 'rb') as f:
    df_feat = pl.read_parquet(f)
cand_id_map = df_feat['candidate_id'].to_list()

print("\n--- Retrieval Statistics ---")
print(f"Recent Index (k=1500) Retrieved: {len(recent_set)}")
print(f"Last Two Index (k=1000) Retrieved: {len(last_two_set)}")
print(f"Full Index (k=500) Retrieved: {len(full_set)}")
print(f"Union Size: {union_size}")

print("\n--- Overlap Percentages (of Union Size) ---")
print(f"All three indices: {len(all_three)} ({len(all_three)/union_size*100:.1f}%)")
print(f"Recent + Last Two (intersection): {len(recent_and_last_two)} ({len(recent_and_last_two)/union_size*100:.1f}%)")
print(f"Recent ONLY: {len(recent_only)} ({len(recent_only)/union_size*100:.1f}%)")
print(f"Full ONLY: {len(full_only)} ({len(full_only)/union_size*100:.1f}%)")

print("\n--- Similarity Score Ranges ---")
for idx_name, k in configs:
    r = results[idx_name]
    print(f"{idx_name}: [Min: {r['min_score']:.4f}, Max: {r['max_score']:.4f}]")

print("\n--- Top 20 Candidate IDs ---")
for idx_name, k in configs:
    r = results[idx_name]
    top_20_faiss = r['ids'][:20]
    top_20_cands = [cand_id_map[i] for i in top_20_faiss]
    print(f"\n{idx_name} Top 20:")
    print(", ".join(top_20_cands))

