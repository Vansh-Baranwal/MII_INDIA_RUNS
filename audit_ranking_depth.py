import polars as pl
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from collections import Counter

artifacts_dir = Path(r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS\artifacts")

# Reconstruct the union pool
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

# Also load parsed candidates to get full_profile_text for keyword analysis
df_parsed = pl.read_parquet(artifacts_dir / 'parsed_candidates.parquet', glob=False)

# Filter to feat_ranking_depth >= 0.90
high_rd = df_pool.filter(pl.col("feat_ranking_depth") >= 0.90)
high_rd_ids = high_rd['candidate_id'].to_list()

# Join with parsed to get full text
high_rd_with_text = high_rd.join(
    df_parsed.select(['candidate_id', 'full_profile_text']),
    on='candidate_id', how='left', suffix='_parsed'
)

# RANKING_TERMS from feature_engineering.py
RANKING_TERMS = {
    "learning-to-rank": 1.0, "lambdamart": 1.0, "re-ranking": 0.9,
    "recommendation systems": 0.8, "personalization": 0.8, "collaborative filtering": 0.7,
    "xgboost": 0.7, "lightgbm": 0.7
}

# Persona classification keywords
search_kw = ['search engineer', 'search ranking', 'information retrieval', 'ir engineer', 'relevance engineer']
recsys_kw = ['recommendation', 'recsys', 'rec sys']
genai_kw = ['rag', 'retrieval augmented', 'langchain', 'llm', 'generative ai', 'genai', 'chatbot']
prompt_kw = ['prompt engineer', 'prompt design', 'prompt tuning']
cv_kw = ['computer vision', 'image', 'opencv', 'yolo', 'object detection']

print(f"Total candidates in pool: {len(df_pool)}")
print(f"Candidates with feat_ranking_depth >= 0.90: {len(high_rd)}")

print(f"\n{'='*130}")
print(f"{'ID':<16} {'Title':<35} {'YoE':>5} {'RankD':>7} {'Builder':>8} {'SR':>7} {'RetD':>7}")
print(f"{'='*130}")

titles = []
persona_map = {}

for row in high_rd_with_text.sort("feat_ranking_depth", descending=True).iter_rows(named=True):
    cid = row['candidate_id']
    title = str(row.get('current_title', 'Unknown'))
    yoe = row.get('years_of_experience', 0)
    rd = row['feat_ranking_depth']
    bs = row['feat_builder_score']
    sr = row['feat_search_relevance_evidence']
    ret = row['feat_retrieval_depth']
    
    titles.append(title)
    
    txt = str(row.get('full_profile_text', '') or '').lower()
    title_lower = title.lower()
    
    # Classify persona
    persona = 'Generic ML'
    if any(k in title_lower for k in search_kw) or any(k in title_lower for k in ['search', 'ranking', 'relevance']):
        persona = 'Search/Ranking'
    elif any(k in title_lower for k in recsys_kw):
        persona = 'Recommendation'
    elif any(k in title_lower for k in ['computer vision', 'cv engineer', 'vision']):
        persona = 'Computer Vision'
    elif any(k in title_lower for k in ['prompt', 'genai', 'generative']):
        persona = 'Prompt/GenAI'
    elif any(k in title_lower for k in ['nlp', 'natural language']):
        persona = 'NLP'
    elif any(k in title_lower for k in ['data scientist']):
        persona = 'Data Scientist'
    elif any(k in title_lower for k in ['research']):
        persona = 'Research'
    
    # Check text for deeper persona signals
    text_search = sum(1 for k in ['learning-to-rank', 'lambdamart', 'ndcg', 'ranking model', 'search ranking', 'query', 'relevance'] if k in txt)
    text_recsys = sum(1 for k in ['recommendation system', 'collaborative filtering', 'matrix factorization', 'two-tower', 'recsys'] if k in txt)
    text_genai = sum(1 for k in ['rag', 'langchain', 'llm', 'chatbot', 'prompt engineering', 'generative ai'] if k in txt)
    text_cv = sum(1 for k in ['computer vision', 'opencv', 'yolo', 'image classification', 'object detection'] if k in txt)
    
    # Count which RANKING_TERMS actually matched
    matched_terms = []
    for term in RANKING_TERMS:
        if term in txt:
            matched_terms.append(term)
    
    persona_map[cid] = {
        'persona': persona, 'text_search': text_search, 'text_recsys': text_recsys,
        'text_genai': text_genai, 'text_cv': text_cv, 'matched_terms': matched_terms
    }
    
    print(f"{cid:<16} {title:<35} {yoe:5.1f} {rd:7.3f} {bs:8.3f} {sr:7.3f} {ret:7.3f}  [{persona}] terms={matched_terms}")

# --- Aggregates ---
print(f"\n{'='*70}")
print("1. Title Distribution")
print(f"{'='*70}")
title_counts = Counter(titles)
for t, c in title_counts.most_common():
    print(f"  {t}: {c}")

bs_vals = high_rd['feat_builder_score'].to_numpy().astype(float)
ret_vals = high_rd['feat_retrieval_depth'].to_numpy().astype(float)
sr_vals = high_rd['feat_search_relevance_evidence'].to_numpy().astype(float)
rd_vals = high_rd['feat_ranking_depth'].to_numpy().astype(float)
yoe_vals = high_rd['years_of_experience'].to_numpy().astype(float)

print(f"\n{'='*70}")
print("2. Averages for feat_ranking_depth >= 0.90 cohort")
print(f"{'='*70}")
print(f"  Avg Builder Score:      {np.mean(bs_vals):.3f} (median: {np.median(bs_vals):.3f})")
print(f"  Avg Retrieval Depth:    {np.mean(ret_vals):.3f} (median: {np.median(ret_vals):.3f})")
print(f"  Avg Search Relevance:   {np.mean(sr_vals):.3f} (median: {np.median(sr_vals):.3f})")
print(f"  Avg Ranking Depth:      {np.mean(rd_vals):.3f} (median: {np.median(rd_vals):.3f})")
print(f"  Avg Years Experience:   {np.mean(yoe_vals):.1f} (median: {np.median(yoe_vals):.1f})")

# --- Persona breakdown ---
persona_counter = Counter(p['persona'] for p in persona_map.values())
print(f"\n{'='*70}")
print("3. Persona Distribution")
print(f"{'='*70}")
for p, c in persona_counter.most_common():
    pct = c / len(persona_map) * 100
    print(f"  {p}: {c} ({pct:.1f}%)")

# --- Term frequency analysis ---
print(f"\n{'='*70}")
print("4. Which RANKING_TERMS are actually triggering high scores?")
print(f"{'='*70}")
term_freq = Counter()
for p in persona_map.values():
    for t in p['matched_terms']:
        term_freq[t] += 1

for t, c in term_freq.most_common():
    pct = c / len(persona_map) * 100
    print(f"  '{t}': found in {c}/{len(persona_map)} candidates ({pct:.1f}%)")

# --- Deep dive: are these candidates keyword-stuffing xgboost/lightgbm? ---
print(f"\n{'='*70}")
print("5. Keyword Stuffing Analysis")
print(f"{'='*70}")

# For each candidate, count how many distinct ranking terms matched vs total count
for row in high_rd_with_text.sort("feat_ranking_depth", descending=True).iter_rows(named=True):
    cid = row['candidate_id']
    txt = str(row.get('full_profile_text', '') or '').lower()
    
    total_hits = 0
    term_breakdown = {}
    for term, w in RANKING_TERMS.items():
        c = txt.count(term)
        if c > 0:
            term_breakdown[term] = c
            total_hits += c * w
    
    raw_score = total_hits
    normalized = min(raw_score / 3.0, 1.0)
    
    # Check if score is dominated by a single generic term
    dominant = max(term_breakdown.items(), key=lambda x: x[1]) if term_breakdown else ("none", 0)
    
    print(f"  {cid} | raw={raw_score:.1f} norm={normalized:.3f} | breakdown={term_breakdown} | dominant='{dominant[0]}' ({dominant[1]}x)")

# --- False positive check ---
print(f"\n{'='*70}")
print("6. Candidates whose ranking_depth is driven ONLY by xgboost/lightgbm")
print(f"{'='*70}")
xgb_only = 0
for row in high_rd_with_text.iter_rows(named=True):
    cid = row['candidate_id']
    txt = str(row.get('full_profile_text', '') or '').lower()
    
    has_true_ranking = any(t in txt for t in ['learning-to-rank', 'lambdamart', 're-ranking'])
    has_recsys = any(t in txt for t in ['recommendation systems', 'collaborative filtering', 'personalization'])
    has_xgb = 'xgboost' in txt or 'lightgbm' in txt
    
    if has_xgb and not has_true_ranking and not has_recsys:
        xgb_only += 1
        title = row.get('current_title', '')
        print(f"  FALSE POSITIVE: {cid} ({title}) — ranking_depth driven purely by xgboost/lightgbm mention")

print(f"\n  Total false positives (xgb/lgb only): {xgb_only} / {len(high_rd)} ({xgb_only/len(high_rd)*100:.1f}%)")
