import polars as pl
from pathlib import Path
import re

artifacts_dir = Path(r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS\artifacts")
df_debug = pl.read_parquet(artifacts_dir / 'debug_top200.parquet', glob=False)

old_top20_ids = [
    'CAND_0064326', 'CAND_0043860', 'CAND_0010685', 'CAND_0046132', 'CAND_0010770', 
    'CAND_0017178', 'CAND_0080534', 'CAND_0065786', 'CAND_0030031', 'CAND_0050454', 
    'CAND_0054394', 'CAND_0002025', 'CAND_0081846', 'CAND_0062247', 'CAND_0043381', 
    'CAND_0060054', 'CAND_0086022', 'CAND_0078042', 'CAND_0046064', 'CAND_0079387'
]

df_top20 = df_debug.head(20)
new_top20_ids = df_top20['candidate_id'].to_list()

leaving = set(old_top20_ids) - set(new_top20_ids)
entering = set(new_top20_ids) - set(old_top20_ids)

print("--- Leaving Top 20 ---")
df_leaving = df_debug.filter(pl.col('candidate_id').is_in(list(leaving)))
for row in df_leaving.iter_rows(named=True):
    print(f"{row['candidate_id']} | {row['current_title']} | Builder: {row['feat_builder_score']}")

print("\n--- Entering Top 20 ---")
df_entering = df_debug.filter(pl.col('candidate_id').is_in(list(entering)))
for row in df_entering.iter_rows(named=True):
    print(f"{row['candidate_id']} | {row['current_title']} | Builder: {row['feat_builder_score']}")

print("\n--- New Top 20 ---")
for i, row in enumerate(df_top20.iter_rows(named=True)):
    print(f"{i+1}. {row['candidate_id']} | {row['current_title']} | Score: {row['Final_Score']:.4f} | Builder: {row['feat_builder_score']} | YoE: {row['years_of_experience']}")

# Counts
titles = df_top20['current_title'].fill_null("").str.to_lowercase().to_list()
c_search = sum(1 for t in titles if 'search' in t)
c_rec = sum(1 for t in titles if 'recommendation' in t)
c_nlp = sum(1 for t in titles if 'nlp' in t)
c_res = sum(1 for t in titles if 'research' in t)
c_jun = sum(1 for t in titles if 'junior' in t or 'associate' in t)

print("\n--- Title Counts ---")
print(f"Search Engineers: {c_search}")
print(f"Recommendation Engineers: {c_rec}")
print(f"NLP Engineers: {c_nlp}")
print(f"Research Engineers: {c_res}")
print(f"Junior titles: {c_jun}")

# Averages
df_old = df_debug.filter(pl.col('candidate_id').is_in(old_top20_ids))
old_b_avg = df_old['feat_builder_score'].mean()
new_b_avg = df_top20['feat_builder_score'].mean()

old_yoe_avg = df_old['years_of_experience'].mean()
new_yoe_avg = df_top20['years_of_experience'].mean()

print("\n--- Averages ---")
print(f"Average Builder Score (Old): {old_b_avg:.3f}")
print(f"Average Builder Score (New): {new_b_avg:.3f}")
print(f"Average YoE (Old): {old_yoe_avg:.2f}")
print(f"Average YoE (New): {new_yoe_avg:.2f}")
