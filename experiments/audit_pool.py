import faiss
import numpy as np
import polars as pl
from pathlib import Path
from sentence_transformers import SentenceTransformer
import re

artifacts_dir = Path(r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS\artifacts")
base_dir = artifacts_dir.parent.parent

# 1. Embed JD
jd_txt_path = base_dir / 'read_docx_output.txt'
if jd_txt_path.exists():
    with open(jd_txt_path, 'r', encoding='utf-8') as f:
        jd_text = f.read()
else:
    jd_text = "Senior AI Engineer Search Ranking Retrieval Embeddings NDCG Vector Databases"

model = SentenceTransformer('all-MiniLM-L6-v2')
jd_emb = model.encode([jd_text])
faiss.normalize_L2(jd_emb)

# 2. Query indices
configs = [
    ('candidates_recent.faiss', 1500),
    ('candidates_last_two.faiss', 1000),
    ('candidates_full.faiss', 500)
]

retrieved_ids = {}
results = {}

with open(artifacts_dir / 'features.parquet', 'rb') as f:
    df_feat = pl.read_parquet(f)
cand_id_map = df_feat['candidate_id'].to_list()

for idx_name, k in configs:
    idx_path = artifacts_dir / idx_name
    index = faiss.read_index(str(idx_path))
    D, I = index.search(jd_emb, k)
    ids = I[0].tolist()
    cands = [cand_id_map[i] for i in ids]
    retrieved_ids[idx_name] = set(cands)
    results[idx_name] = cands

union_set = retrieved_ids['candidates_recent.faiss'] | retrieved_ids['candidates_last_two.faiss'] | retrieved_ids['candidates_full.faiss']

# Load parsed candidates
df_parsed = pl.read_parquet(artifacts_dir / 'parsed_candidates.parquet', glob=False)

# Filter for union pool
df_union = df_parsed.filter(pl.col('candidate_id').is_in(list(union_set)))

print(f"Union Pool Size: {len(df_union)}")

# Top 50 extraction
for idx_name, _ in configs:
    print(f"\n=========================================")
    print(f"Top 50 from {idx_name}")
    print(f"=========================================")
    top_50_ids = results[idx_name][:50]
    
    # Maintain order
    df_top50 = pl.DataFrame({"candidate_id": top_50_ids}).join(df_union, on="candidate_id", how="left")
    
    # Safely extract fields handling nulls
    for row in df_top50.iter_rows(named=True):
        title = row.get('current_title', 'Unknown')
        yoe = row.get('total_years_experience', 'Unknown')
        ind = row.get('current_industry', 'Unknown')
        comp_size = row.get('company_size', 'Unknown')
        print(f"[{row['candidate_id']}] Title: {title} | YoE: {yoe} | Ind: {ind} | Size: {comp_size}")

# Compute counts over union pool
titles = df_union['current_title'].fill_null("").str.to_lowercase().to_list()

def count_matches(pattern):
    return sum(1 for t in titles if re.search(pattern, t))

cnt_research = count_matches(r'research')
cnt_architect = count_matches(r'architect')
cnt_manager = count_matches(r'manager|director|vp|head')
cnt_ds = count_matches(r'data scientist')
cnt_search = count_matches(r'search|relevance|retrieval')
cnt_rec = count_matches(r'recommendation|recsys')

print(f"\n--- Union Pool Role Breakdown ({len(titles)} total) ---")
print(f"Research titles: {cnt_research}")
print(f"Architect titles: {cnt_architect}")
print(f"Manager titles: {cnt_manager}")
print(f"Data Scientist titles: {cnt_ds}")
print(f"Search/Relevance titles: {cnt_search}")
print(f"Recommendation titles: {cnt_rec}")

# Identify false positives
cnt_educator = count_matches(r'teacher|professor|lecturer|instructor|faculty')
cnt_consultant = count_matches(r'consultant|freelance|adviser|advisor')
cnt_wrapper = count_matches(r'prompt engineer|openai|chatgpt') # Proxy for AI wrappers
cnt_obvious_fp = count_matches(r'sales|recruiter|hr|marketing|finance|accountant')

print(f"\n--- Potential Red Flags ---")
print(f"Obvious false positives (HR/Sales/Finance): {cnt_obvious_fp}")
print(f"Likely educators (Teacher/Prof): {cnt_educator}")
print(f"Likely consultants (Freelance/Advisor): {cnt_consultant}")
print(f"Likely Wrapper-AI (Prompt/ChatGPT): {cnt_wrapper}")

# Est
relevant = len(titles) - (cnt_obvious_fp + cnt_educator + cnt_consultant + cnt_wrapper)
perc_relevant = (relevant / len(titles)) * 100
perc_penalty = ((cnt_obvious_fp + cnt_educator + cnt_consultant + cnt_wrapper) / len(titles)) * 100

print(f"\n--- Estimates ---")
print(f"Likely relevant pool: {perc_relevant:.1f}%")
print(f"Requires reranking penalties: {perc_penalty:.1f}%")
