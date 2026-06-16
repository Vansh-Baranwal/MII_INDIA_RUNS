import numpy as np
import faiss
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    logging.info(f"Creating IndexFlatIP with dimension {dim}")
    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def main():
    artifacts_dir = Path(__file__).resolve().parent.parent.parent / 'artifacts'
    
    for prefix in ['recent', 'last_two', 'full']:
        emb_file = artifacts_dir / f'embeddings_{prefix}.npy'
        if not emb_file.exists():
            logging.warning(f"File {emb_file} not found, skipping...")
            continue
            
        logging.info(f"Loading {emb_file}")
        embeddings = np.load(emb_file)
        
        index = create_faiss_index(embeddings)
        index_file = artifacts_dir / f'candidates_{prefix}.faiss'
        
        logging.info(f"Writing index to {index_file}")
        faiss.write_index(index, str(index_file))
        
    logging.info("FAISS index building complete.")

if __name__ == "__main__":
    main()
