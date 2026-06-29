import polars as pl
from pathlib import Path
import json

base_dir = Path(r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS")

df_top = pl.read_csv(base_dir / 'top100.csv', glob=False).head(50)
df_parsed = pl.read_parquet(base_dir / 'artifacts/parsed_candidates.parquet', glob=False)

df = df_top.join(df_parsed, on="candidate_id", how="left")

out = []
for i, row in enumerate(df.to_dicts()):
    cid = row['candidate_id']
    rank = i + 1
    title = row.get('current_title', '')
    yoe = row.get('years_of_experience', 0)
    
    # parse skills
    skills_str = ""
    try:
        skills = json.loads(row.get('skills_json', '[]'))
        s_names = [s.get('name', '') for s in skills if s.get('name')]
        skills_str = ", ".join(s_names[:15]) # top 15 skills
    except:
        pass
        
    # parse work history
    hist_str = ""
    try:
        hist = json.loads(row.get('work_history_json', '[]'))
        h_titles = []
        for h in hist:
            t = h.get('title', '')
            desc = h.get('description', '')[:50].replace('\n', ' ')
            h_titles.append(f"{t} ({desc}...)")
        hist_str = " | ".join(h_titles[:3]) # last 3 jobs
    except:
        pass
        
    out.append(f"Rank {rank}: {cid} | {title} | {yoe} YOE\nSkills: {skills_str}\nHistory: {hist_str}\n---")
    
with open(base_dir / 'top50_raw_text.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print("Extraction complete.")
