import polars as pl
import numpy as np
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_embeddings(texts, model, batch_size=256):
    logging.info(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings

def main():
    artifacts_dir = Path(__file__).resolve().parent.parent.parent / 'artifacts'
    input_file = artifacts_dir / 'features.parquet'
    
    if not input_file.exists():
        logging.error(f"{input_file} not found.")
        return
        
    logging.info("Loading candidates...")
    df = pl.read_parquet(input_file)
    
    # Ensure columns exist
    if not all(col in df.columns for col in ['recent_role_text', 'last_two_roles_text', 'full_profile_text']):
        logging.error("Missing required text columns in parquet.")
        return
        
    recent_texts = df['recent_role_text'].to_list()
    last_two_texts = df['last_two_roles_text'].to_list()
    full_texts = df['full_profile_text'].to_list()
    
    logging.info("Loading sentence-transformer model (all-MiniLM-L6-v2)")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    emb_recent = generate_embeddings(recent_texts, model)
    emb_last_two = generate_embeddings(last_two_texts, model)
    emb_full = generate_embeddings(full_texts, model)
    
    logging.info("Saving numpy arrays")
    np.save(artifacts_dir / 'embeddings_recent.npy', emb_recent)
    np.save(artifacts_dir / 'embeddings_last_two.npy', emb_last_two)
    np.save(artifacts_dir / 'embeddings_full.npy', emb_full)
    
    logging.info("Embeddings generation complete.")

if __name__ == "__main__":
    main()
