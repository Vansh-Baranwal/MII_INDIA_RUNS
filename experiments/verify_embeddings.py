import numpy as np
import os

files = [
    r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS\artifacts\embeddings_recent.npy",
    r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS\artifacts\embeddings_last_two.npy",
    r"d:\Some_stuffs\India Runs\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\INDIA_RUNS\artifacts\embeddings_full.npy"
]

for p in files:
    name = os.path.basename(p)
    print(f"--- {name} ---")
    arr = np.load(p)
    
    shape = arr.shape
    dtype = arr.dtype
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    # Calculate norms along axis 1
    norms = np.linalg.norm(arr, axis=1)
    mean_norm = np.mean(norms)
    
    nan_count = np.sum(np.isnan(arr))
    inf_count = np.sum(np.isinf(arr))
    
    all_zeros = np.all(arr == 0)
    
    print(f"Shape: {shape}")
    print(f"Dtype: {dtype}")
    print(f"Min Value: {min_val:.6f}")
    print(f"Max Value: {max_val:.6f}")
    print(f"Mean Norm: {mean_norm:.6f}")
    print(f"NaN Count: {nan_count}")
    print(f"Inf Count: {inf_count}")
    print(f"Valid Shape (100000, 384): {shape == (100000, 384)}")
    print(f"All Finite: {nan_count == 0 and inf_count == 0}")
    print(f"Not All Zeros: {not all_zeros}")
    print()
