import polars as pl
import numpy as np
import faiss
import math
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from reasoning import generate_reasoning

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    artifacts_dir = Path(__file__).resolve().parent.parent.parent / 'artifacts'
    jd_path = base_dir / 'job_description.docx' # assuming we converted it or read text, for simplicity using markdown if exists
    
    # Check if a text version exists, else we might have extracted it earlier
    jd_txt_path = base_dir / 'read_docx_output.txt'
    if jd_txt_path.exists():
        jd_text = load_text(jd_txt_path)
    else:
        jd_text = "Senior AI Engineer Search Ranking Retrieval Embeddings NDCG Vector Databases" # fallback
        
    logging.info("Encoding JD...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    jd_emb = model.encode([jd_text])
    faiss.normalize_L2(jd_emb)
    
    indices_set = set()
    retrieval_configs = [
        ('recent', 1500),
        ('last_two', 1000),
        ('full', 500)
    ]
    
    for prefix, k in retrieval_configs:
        idx_path = artifacts_dir / f'candidates_{prefix}.faiss'
        if idx_path.exists():
            logging.info(f"Searching {idx_path} for top {k}...")
            index = faiss.read_index(str(idx_path))
            _, I = index.search(jd_emb, k)
            indices_set.update(I[0].tolist())
            
    logging.info(f"Union pool size: {len(indices_set)}")
    if not indices_set:
        logging.error("No candidates retrieved. Exiting.")
        return
        
    logging.info("Loading parquet features...")
    df = pl.read_parquet(artifacts_dir / 'features.parquet')
    
    # We assume 'candidate_index' or row number was preserved. 
    # For this implementation, we will use row numbers as indices to match FAISS
    df = df.with_row_index("faiss_id")
    df_pool = df.filter(pl.col("faiss_id").is_in(list(indices_set)))
    
    # Load embeddings to compute precise cosine similarity for the pool
    emb_recent = np.load(artifacts_dir / 'embeddings_recent.npy')
    emb_last_two = np.load(artifacts_dir / 'embeddings_last_two.npy')
    emb_full = np.load(artifacts_dir / 'embeddings_full.npy')
    
    faiss.normalize_L2(emb_recent)
    faiss.normalize_L2(emb_last_two)
    faiss.normalize_L2(emb_full)
    
    pool_ids = df_pool['faiss_id'].to_list()
    
    sim_recent = (emb_recent[pool_ids] @ jd_emb[0]).tolist()
    sim_last_two = (emb_last_two[pool_ids] @ jd_emb[0]).tolist()
    sim_full = (emb_full[pool_ids] @ jd_emb[0]).tolist()
    
    df_pool = df_pool.with_columns([
        pl.Series("sim_recent", sim_recent),
        pl.Series("sim_last_two", sim_last_two),
        pl.Series("sim_full", sim_full),
    ])
    
    df_pool = df_pool.with_columns(
        Base_Score = (0.55 * pl.col("sim_recent")) + (0.30 * pl.col("sim_last_two")) + (0.15 * pl.col("sim_full"))
    )
    
    # Final Math
    df_pool = df_pool.with_columns(
        Technical_Multiplier = 1.0 + (
            0.25 * pl.col("feat_search_relevance_evidence") +
            0.20 * pl.col("feat_retrieval_depth") +
            0.20 * pl.col("feat_evaluation_rigor") +
            0.15 * pl.col("feat_builder_score")
        ),
        Trajectory_Multiplier = pl.col("feat_product_exposure") + pl.col("feat_trajectory_transition"),
        Behavioral_Multiplier = pl.col("feat_availability_score"),
        Persona_Penalty = pl.col("feat_wrapper_ai_only") * pl.col("feat_architect_no_coding"),
        Honeypot_Decay = pl.col("contradiction_score").map_elements(lambda x: math.exp(-x), return_dtype=pl.Float64)
    )
    
    df_pool = df_pool.with_columns(
        Final_Score = (
            pl.col("Base_Score") *
            pl.col("Technical_Multiplier") *
            pl.col("Trajectory_Multiplier") *
            pl.col("Behavioral_Multiplier") *
            pl.col("Persona_Penalty") *
            pl.col("Honeypot_Decay")
        )
    )
    
    # Top 100
    df_top = df_pool.sort(["Final_Score", "candidate_id"], descending=[True, False]).head(100)
    
    logging.info("Generating reasoning...")
    reasoning_list = []
    for row in df_top.to_dicts():
        reasoning_list.append(generate_reasoning(row))
        
    df_top = df_top.with_columns(pl.Series("reasoning", reasoning_list))
    
    # Submission Format
    df_sub = df_top.select([
        pl.col("candidate_id"),
        pl.arange(1, 101).alias("rank"),
        pl.col("Final_Score").alias("score"),
        pl.col("reasoning")
    ])
    
    sub_path = base_dir / 'submission.csv'
    df_sub.write_csv(sub_path)
    logging.info(f"Saved submission to {sub_path}")
    
    # Debug Parquet
    debug_path = artifacts_dir / 'candidate_debug.parquet'
    df_pool.sort(["Final_Score", "candidate_id"], descending=[True, False]).head(200).write_parquet(debug_path)
    logging.info(f"Saved debug output to {debug_path}")

if __name__ == "__main__":
    main()
