import polars as pl
from pathlib import Path
base_dir = Path(r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS")
df = pl.read_csv(base_dir / 'top500.csv', glob=False)
df_slice = df.slice(89, 31) # Ranks 90 to 120 (indices 89 to 119)
df_parsed = pl.read_parquet(base_dir / 'artifacts/parsed_candidates.parquet', glob=False)
df_slice = df_slice.join(df_parsed, on="candidate_id", how="left")

import json
for i, row in enumerate(df_slice.to_dicts()):
    try:
        skills = json.loads(row.get('skills_json', '[]'))
        s_names = [s.get('name', '') for s in skills if s.get('name')]
        skills_str = ", ".join(s_names[:10])
    except:
        skills_str = ""
    print(f"Rank {89 + i + 1}: {row['candidate_id']} | {row.get('current_title','')} ({row.get('years_of_experience',0)} YOE) | Final Score: {row['Final_Score']:.3f} | Skills: {skills_str}")
