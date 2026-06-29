import polars as pl
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_pipeline(artifacts_dir: Path):
    features_path = artifacts_dir / 'features.parquet'
    submission_path = artifacts_dir.parent / 'submission.csv'
    
    if not features_path.exists():
        logging.error("features.parquet not found.")
        return
        
    logging.info("--- 1. Loading Features ---")
    df = pl.read_parquet(features_path, glob=False)
    logging.info(f"Loaded {len(df)} candidates.")
    
    logging.info("\n--- 2. Checking for NaNs ---")
    null_counts = df.null_count().to_dicts()[0]
    for col, count in null_counts.items():
        if count > 0:
            logging.warning(f"NaN Detected: Column '{col}' has {count} nulls.")
    
    logging.info("\n--- 3. Checking Constant-Value Features ---")
    for col in df.columns:
        if df[col].n_unique() == 1:
            logging.warning(f"Constant Feature Detected: '{col}' has only 1 unique value.")
            
    logging.info("\n--- 4. Feature Distributions & Anomalies ---")
    numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
    
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        mean_val = df[col].mean()
        logging.info(f"Feature '{col}': Min={min_val:.4f}, Max={max_val:.4f}, Mean={mean_val:.4f}")
        
        # Detect explosions
        if max_val > 100 and col not in ['grad_year', 'total_duration_months', 'candidate_id', 'expected_salary_min', 'expected_salary_max']:
            logging.warning(f"Suspicious Score Explosion in '{col}': Max value is {max_val:.2f}")
            
    logging.info("\n--- 5. Contradiction Score Anomalies ---")
    contradictions = df['contradiction_score']
    high_contradictions = contradictions.filter(contradictions > 5.0)
    logging.info(f"Found {len(high_contradictions)} candidates with severe contradiction scores (>5.0).")
    if len(high_contradictions) == 0:
        logging.warning("No severe contradictions found. Honeypot logic might be too weak.")
        
    logging.info("\n--- 6. Submission.csv Validity ---")
    if submission_path.exists():
        sub_df = pl.read_csv(submission_path, glob=False)
        logging.info(f"Loaded submission.csv with {len(sub_df)} rows.")
        
        if len(sub_df) != 100:
            logging.error(f"Submission length is {len(sub_df)}, expected 100.")
            
        expected_cols = ['candidate_id', 'rank', 'score', 'reasoning']
        for col in expected_cols:
            if col not in sub_df.columns:
                logging.error(f"Missing required column '{col}' in submission.csv")
                
        # Check Monotonicity
        scores = sub_df['score'].to_list()
        is_monotonic = all(x >= y for x, y in zip(scores, scores[1:]))
        if not is_monotonic:
            logging.error("Ranking Monotonicity FAILED! Scores are not strictly decreasing.")
        else:
            logging.info("Ranking Monotonicity PASSED.")
            
        ranks = sub_df['rank'].to_list()
        if ranks != list(range(1, 101)):
            logging.error("Ranks are not exactly 1 to 100.")
        else:
            logging.info("Ranks 1-100 PASSED.")
            
        # Check Reasoning Variance
        unique_reasonings = sub_df['reasoning'].n_unique()
        logging.info(f"Unique reasoning strings: {unique_reasonings}/100")
        if unique_reasonings < 10:
            logging.warning("Reasoning generation lacks variance. (Found < 10 unique strings)")
            
    else:
        logging.error("submission.csv not found.")

if __name__ == "__main__":
    artifacts_dir = Path(__file__).resolve().parent.parent.parent / 'artifacts'
    evaluate_pipeline(artifacts_dir)
