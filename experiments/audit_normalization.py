import polars as pl
from collections import defaultdict
from pathlib import Path

artifacts_dir = Path(r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS\artifacts")

# Original Dictionary
orig = {
    "candidate matching": 1.0, "talent discovery": 1.0, "relevance optimization": 1.0,
    "marketplace ranking": 1.0, "query understanding": 1.0, "semantic search": 1.0,
    "inverted index": 0.8, "bm25": 0.8, "lucene": 0.7, "solr": 0.7, "tf-idf": 0.4
}

# Expansions
expansions = {
    "candidate matching": ["candidate matching", "candidate matcher", "talent matching", "candidate recommendation"],
    "talent discovery": ["talent discovery", "talent search", "people search", "candidate search"],
    "relevance optimization": ["relevance optimization", "ranking optimization", "search ranking", "ranking system", "search relevance", "search quality"],
    "marketplace ranking": ["marketplace ranking", "marketplace search"],
    "query understanding": ["query understanding", "query intent", "intent modeling", "search intent", "query expansion"],
    "semantic search": ["semantic search", "vector search", "dense search", "neural search", "dense retrieval"],
    "inverted index": ["inverted index", "inverted indexes", "inverted-index", "inverted_index"],
    "bm25": ["bm25", "okapi bm25", "bm-25", "bm 25"],
    "lucene": ["lucene", "apache lucene", "lucene/solr"],
    "solr": ["solr", "apache solr"],
    "tf-idf": ["tf-idf", "tfidf", "tf idf"]
}

# Missed Terms Pools
search_pool = ["elasticsearch", "opensearch", "vespa", "weaviate", "qdrant", "milvus", "pinecone", "chroma", "faiss", "hnsw", "ann", "approximate nearest neighbor", "lexical search", "keyword search", "full-text search", "full text search", "text retrieval", "document retrieval", "bi-encoder", "cross-encoder", "colbert", "rag", "retrieval augmented generation", "vector database"]
ranking_pool = ["learning to rank", "learning-to-rank", "ltr", "lambdamart", "lambdarank", "ranknet", "xgboost", "lightgbm", "catboost", "listwise", "pairwise", "pointwise", "click model", "position bias", "ndcg", "dcg", "mean reciprocal rank", "mrr", "re-ranking", "reranking"]
recsys_pool = ["recommendation system", "recsys", "collaborative filtering", "matrix factorization", "two-tower", "two tower", "dssm", "wide and deep", "deepfm", "content-based filtering", "personalization", "item-item", "user-item", "session-based recommendation"]

print("Loading data...")
df_all = pl.read_parquet(artifacts_dir / 'parsed_candidates.parquet', glob=False)

exact_counts = defaultdict(int)
exp_counts = defaultdict(int)

search_counts = defaultdict(int)
ranking_counts = defaultdict(int)
recsys_counts = defaultdict(int)

# Dictionary for Top 30 analysis
# I will load the exact Top 30 from Scenario D. 
# For simplicity, we just use the known Top 30 candidate IDs or re-compute them quickly.
# I'll just use the union pool features and recompute the final score.
import numpy as np
import math
df_features = pl.read_parquet(artifacts_dir / 'features.parquet', glob=False)
df_features = df_features.with_columns(
    Behavioral_Multiplier = 1.0 + (pl.col("feat_availability_score") + pl.col("feat_saved_boost") + pl.col("feat_search_appearance_boost") - 1.0) * 0.25,
    Technical_Multiplier = 1.0 + (0.25 * pl.col("feat_search_relevance_evidence") + 0.20 * pl.col("feat_ranking_depth") + 0.15 * pl.col("feat_retrieval_depth") + 0.15 * pl.col("feat_evaluation_rigor") + 0.35 * pl.col("feat_builder_score"))
)
# Since I didn't recompute base score here, I'll just pull it from the exact texts if needed.
# Actually, I can just score ALL 100k candidates on the expanded SR and take the top gainers, 
# but the prompt specifically asked for "which candidates in the current Top 30 would gain the most".
# The Top 30 IDs from the previous run:
top30_ids = [
    "CAND_0064326", "CAND_0010685", "CAND_0005649", "CAND_0010770", "CAND_0028793", "CAND_0051292", "CAND_0013613", "CAND_0060054", "CAND_0032515", "CAND_0086022",
    "CAND_0053591", "CAND_0043860", "CAND_0094759", "CAND_0043381", "CAND_0069638", "CAND_0030031", "CAND_0050454", "CAND_0046132", "CAND_0070398", "CAND_0064270",
    "CAND_0078042", "CAND_0068811", "CAND_0065786", "CAND_0042100", "CAND_0030827", "CAND_0030953", "CAND_0062247", "CAND_0075574", "CAND_0000031", "CAND_0050876"
]

top30_gains = []

print("Processing texts...")
for row in df_all.iter_rows(named=True):
    txt = row['full_profile_text'].lower()
    cid = row['candidate_id']
    
    for term in orig:
        if term in txt:
            exact_counts[term] += 1
    
    for term, exps in expansions.items():
        if any(e in txt for e in exps):
            exp_counts[term] += 1
            
    for t in search_pool:
        if t in txt: search_counts[t] += 1
    for t in ranking_pool:
        if t in txt: ranking_counts[t] += 1
    for t in recsys_pool:
        if t in txt: recsys_counts[t] += 1
        
    if cid in top30_ids:
        # Calculate old SR
        old_sr_raw = 0.0
        for term, w in orig.items():
            old_sr_raw += txt.count(term) * w
        old_sr = min(old_sr_raw / 5.0, 1.0)
        
        # Calculate expanded SR (using the combined pools + expanded orig)
        expanded_score = 0.0
        for term, exps in expansions.items():
            w = orig[term]
            # avoid double counting
            matched = False
            for e in exps:
                c = txt.count(e)
                if c > 0:
                    expanded_score += c * w
                    matched = True
        
        for t in search_pool + ranking_pool + recsys_pool:
            c = txt.count(t)
            expanded_score += c * 1.0 # assume weight 1.0 for missed terms
            
        new_sr = min(expanded_score / 5.0, 1.0)
        gain = new_sr - old_sr
        top30_gains.append({
            'candidate_id': cid,
            'title': row['current_title'],
            'old_sr': old_sr,
            'new_sr': new_sr,
            'gain': gain
        })

print("\n--- 1 & 2. Phrase Matching Audit ---")
for term in orig:
    ex = exact_counts[term]
    ex_p = exp_counts[term]
    pct = ((ex_p - ex) / ex * 100) if ex > 0 else (100 if ex_p > 0 else 0)
    print(f"[{term}] Exact: {ex} | Expanded: {ex_p} | Increase: +{pct:.1f}%")

print("\n--- A. Top 20 Missed Search Phrases ---")
for t, c in sorted(search_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
    print(f"- {t}: {c}")

print("\n--- B. Top 20 Missed Ranking Phrases ---")
for t, c in sorted(ranking_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
    print(f"- {t}: {c}")

print("\n--- C. Top 20 Missed Recommendation Phrases ---")
for t, c in sorted(recsys_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
    print(f"- {t}: {c}")

print("\n--- Top Gainers in Top 30 ---")
top30_gains.sort(key=lambda x: x['gain'], reverse=True)
for i, g in enumerate(top30_gains[:20]):
    print(f"{i+1}. {g['candidate_id']} ({g['title']}) | Old SR: {g['old_sr']:.3f} -> New SR: {g['new_sr']:.3f} | Gain: +{g['gain']:.3f}")
