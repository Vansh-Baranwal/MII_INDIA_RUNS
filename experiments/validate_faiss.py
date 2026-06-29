import faiss
import numpy as np
import os
import time

artifacts_dir = r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS\artifacts"
indices = [
    'candidates_recent.faiss',
    'candidates_last_two.faiss',
    'candidates_full.faiss'
]

print("--- FAISS Index Validation ---")
for idx_name in indices:
    idx_path = os.path.join(artifacts_dir, idx_name)
    size = os.path.getsize(idx_path)
    print(f"\nLoading: {idx_name}")
    print(f"File Size: {size} bytes")
    
    try:
        index = faiss.read_index(idx_path)
        print("Successful Load Test: PASS")
        print(f"ntotal (Vectors Indexed): {index.ntotal}")
        print(f"dimension: {index.d}")
        # Index type via class name
        print(f"Index Type: {index.__class__.__name__}")
    except Exception as e:
        print(f"Successful Load Test: FAIL ({e})")

print("\n--- Test Query ---")
# Load the first embedding from recent to use as a test query
emb_path = os.path.join(artifacts_dir, 'embeddings_recent.npy')
test_emb = np.load(emb_path, mmap_mode='r')[0:1].copy()
faiss.normalize_L2(test_emb)

idx_path = os.path.join(artifacts_dir, 'candidates_recent.faiss')
index = faiss.read_index(idx_path)

D, I = index.search(test_emb, 5)

print(f"Test Query executed against candidates_recent.faiss")
print("Top 5 returned IDs (FAISS internal indices):", I[0].tolist())
print("Similarity Scores (Inner Product):", D[0].tolist())
