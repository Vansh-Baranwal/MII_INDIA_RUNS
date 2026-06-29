import polars as pl
import os
import re
from pathlib import Path

base_dir = Path(r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS")
artifacts_dir = base_dir / 'artifacts'

# Paths
sub_path = base_dir / 'submission.csv'
debug_path = artifacts_dir / 'debug_top200.parquet'
top20_path = artifacts_dir / 'top20.csv'
top100_path = artifacts_dir / 'top100.csv'

# Generate missing CSVs if needed
df_debug = pl.read_parquet(debug_path, glob=False)
if not top20_path.exists():
    df_debug.head(20).write_csv(top20_path)
if not top100_path.exists():
    df_debug.head(100).write_csv(top100_path)

# Verify Existence
def file_info(path):
    if not path.exists():
        return "MISSING"
    size = os.path.getsize(path)
    if str(path).endswith('.csv'):
        rows = len(pl.read_csv(path, glob=False))
    else:
        rows = len(pl.read_parquet(path, glob=False))
    return f"Path: {path.name} | Size: {size} bytes | Rows: {rows}"

print("\n--- Generated Artifacts ---")
print(file_info(sub_path))
print(file_info(top20_path))
print(file_info(top100_path))
print(file_info(debug_path))

# Top 20 Audit
print("\n--- Top 20 Audit ---")
df_top20 = df_debug.head(20)
for i, row in enumerate(df_top20.iter_rows(named=True)):
    print(f"Rank {i+1} | ID: {row['candidate_id']} | Score: {row['Final_Score']:.4f} | Title: {row['current_title']} | YoE: {row['years_of_experience']} | Contradiction: {row['contradiction_score']} | Builder: {row['feat_builder_score']:.3f} | SR_Evidence: {row['feat_search_relevance_evidence']:.3f}")

# Quality Review
titles_20 = df_top20['current_title'].fill_null("").str.to_lowercase().to_list()
cnt_search = sum(1 for t in titles_20 if 'search' in t)
cnt_rel = sum(1 for t in titles_20 if 'relevance' in t)
cnt_rec = sum(1 for t in titles_20 if 'recommendation' in t)
cnt_aml = sum(1 for t in titles_20 if 'applied ml' in t or 'applied scientist' in t)
cnt_ds = sum(1 for t in titles_20 if 'data scientist' in t)
cnt_mgr = sum(1 for t in titles_20 if re.search(r'manager|director|vp|head', t))

print("\n--- Top 20 Quality Review: Counts ---")
print(f"Search Engineers: {cnt_search}")
print(f"Relevance Engineers: {cnt_rel}")
print(f"Recommendation Engineers: {cnt_rec}")
print(f"Applied ML Engineers: {cnt_aml}")
print(f"Data Scientists: {cnt_ds}")
print(f"Managers/Directors/VPs: {cnt_mgr}")

# Flags
flags = []
for i, row in enumerate(df_top20.iter_rows(named=True)):
    if row['contradiction_score'] > 0:
        flags.append(f"Rank {i+1} ({row['candidate_id']}) Contradiction: {row['contradiction_score']}")
    if row['feat_builder_score'] < 0.3:
        flags.append(f"Rank {i+1} ({row['candidate_id']}) Low Builder: {row['feat_builder_score']:.3f}")
    if row.get('notice_period_days', 0) and row['notice_period_days'] > 60:
        flags.append(f"Rank {i+1} ({row['candidate_id']}) Notice > 60: {row['notice_period_days']}")
    
    t = row['current_title'].lower() if row['current_title'] else ""
    if re.search(r'sales|recruiter|hr|finance', t):
        flags.append(f"Rank {i+1} ({row['candidate_id']}) False Positive Title: {t}")

print("\n--- Top 20 Quality Review: Flags ---")
if not flags:
    print("None!")
else:
    for f in flags: print(f)

# Top 100 Audit
print("\n--- Top 100 Audit ---")
df_top100 = df_debug.head(100)
titles_100 = df_top100['current_title'].fill_null("").str.to_lowercase().to_list()
cnt_100_search = sum(1 for t in titles_100 if 'search' in t or 'relevance' in t)
cnt_100_aml = sum(1 for t in titles_100 if 'applied' in t)
cnt_100_mgr = sum(1 for t in titles_100 if 'manager' in t or 'director' in t)
print(f"Titles (Search/Rel: {cnt_100_search}, Applied: {cnt_100_aml}, Manager: {cnt_100_mgr})")

yoe_mean = df_top100['years_of_experience'].mean()
yoe_min = df_top100['years_of_experience'].min()
yoe_max = df_top100['years_of_experience'].max()
print(f"YoE Dist: Mean {yoe_mean:.1f}, Min {yoe_min}, Max {yoe_max}")

score_mean = df_top100['Final_Score'].mean()
score_min = df_top100['Final_Score'].min()
score_max = df_top100['Final_Score'].max()
print(f"Score Dist: Mean {score_mean:.3f}, Min {score_min:.3f}, Max {score_max:.3f}")

c_mean = df_top100['contradiction_score'].mean()
c_max = df_top100['contradiction_score'].max()
print(f"Contradiction Dist: Mean {c_mean:.2f}, Max {c_max}")
