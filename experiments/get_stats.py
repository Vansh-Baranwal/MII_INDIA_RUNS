import polars as pl
import os

pl.Config.set_ascii_tables(True)
p = r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS\artifacts\features.parquet"

size = os.path.getsize(p)
with open(p, 'rb') as f:
    df = pl.read_parquet(f)

print(f"Size: {size}")
print(f"Rows: {len(df)}")
print("Schema:")
for col, dtype in df.schema.items():
    print(f"  {col}: {dtype}")

print("\nFirst 5 Rows:")
print(df.select(['candidate_id', 'feat_search_relevance_evidence', 'feat_availability_score', 'feat_saved_boost', 'contradiction_score']).head(5))

print("\nFeature Statistics:")
print(df.select(['feat_search_relevance_evidence', 'feat_availability_score', 'feat_saved_boost', 'contradiction_score']).describe())
