import os
import numpy as np
import pandas as pd
import nibabel as nb
import pooch
from numba import njit, prange
from time import time


GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
REPO_URL = "https://api.github.com/repos/Barafat2/MTB_task_library/contents/"
CACHE_DIR = pooch.os_cache("MTB_task_library")


def fetch_task_library(atlas='fs32k', version='V1'):
    """
    Fetch task activation library from private GitHub repo.

    Args:
        atlas: Atlas name ('fs32k', 'SUIT3', 'MNISymC3')
        version: Library version (e.g., 'V1')

    Returns:
        data: ndarray (n_conditions, n_vertices)
        info: DataFrame with condition info
    """
    if GITHUB_TOKEN is None:
        raise ValueError("GITHUB_TOKEN environment variable not set")

    data_file = f"{version}/desc-tasks{version}_space-{atlas}_beta.dscalar.nii"
    info_file = f"{version}/desc-tasks{version}_info.tsv"

    downloader = pooch.HTTPDownloader(
        headers={"Authorization": f"token {GITHUB_TOKEN}",
                 "Accept": "application/vnd.github.v3.raw"}
    )

    data_path = pooch.retrieve(
        url=REPO_URL + data_file,
        known_hash=None,
        path=CACHE_DIR,
        fname=data_file,
        downloader=downloader
    )
    info_path = pooch.retrieve(
        url=REPO_URL + info_file,
        known_hash=None,
        path=CACHE_DIR,
        fname=info_file,
        downloader=downloader
    )

    data = nb.load(data_path).get_fdata()
    info = pd.read_csv(info_path, sep='\t')

    print(f"Loaded {data.shape[0]} conditions, {data.shape[1]} vertices")
    return data, info

@njit
def _extract_submatrix(G, idx):
    n = len(idx)
    result = np.empty((n, n), dtype=G.dtype)
    for i in range(n):
        for j in range(n):
            result[i, j] = G[idx[i], idx[j]]
    return result


@njit
def _compute_nit(G):
    N = G.shape[0]
    
    # Center the matrix
    H = np.eye(N) - np.ones((N, N)) / N
    G_mc = H @ G @ H
    
    # Eigenvalues
    l_mc = np.linalg.eigvalsh(G_mc)
    
    # Negative inverse trace
    l_mc = l_mc[::-1]
    l_mc[l_mc < 1e-12] = 1e-12
    nit = - np.sum(1 / l_mc)
    return nit


@njit(parallel=True)
def compute_nits_batch(Gs, combinations):
    """
    Compute NIT scores for a batch of combinations.
    
    Args:
        G: Full G matrix (n_conditions x n_conditions)
        combinations: Array of shape (n_combs, k) with condition indices
    
    Returns: Array of NIT scores
    """
    n_combs = combinations.shape[0]
    results = np.empty(n_combs, dtype=np.float64)
    
    for i in prange(n_combs):
        idx = combinations[i]
        comb_g = _extract_submatrix(Gs, idx)
        results[i] = _compute_nit(comb_g)
    
    return results

def get_top_batteries(task_library, condition_labels, n_samples, battery_size=8, n_top_batteries=10, forced_tasks=None):
    """
    Random search over task combinations to find highest NIT.
    
    Args:
        task_library: activation patterns for the library of tasks
        condition_labels: List of condition labels
        n_samples: Total number of random combinations to sample
        battery_size: Number of tasks in each combination
        n_top_batteries: Number of top results to keep
        forced_tasks: List of task labels to include in every combination
    
    Returns: DataFrame with columns 'rank', 'nit', 'tasks'
    """
    n_lib_task = task_library.shape[0]
    results_list = []

    G = task_library @ task_library.T
    
    # Handle forced tasks - convert labels to indices
    if forced_tasks is None:
        forced_tasks = []
    label_to_idx = {label: i for i, label in enumerate(condition_labels)}
    forced_indices = np.array([label_to_idx[t] for t in forced_tasks], dtype=np.int64)
    n_forced = len(forced_indices)
    n_random = battery_size - n_forced
    
    # Available tasks for random selection (exclude forced)
    available_tasks = np.array([i for i in range(n_lib_task) if i not in forced_indices])
    
    # auto choose a batch size based on n_samples, but dont want it to be too big or too small
    batch_size = max(10000, min(250000, n_samples // 10))

    print(f"Starting computation... (forced: {n_forced}, random: {n_random})")
    start_time = time()
    
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_n = batch_end - batch_start
        
        # Generate combinations
        combinations = np.empty((batch_n, battery_size), dtype=np.int64)
        for i in range(batch_n):
            random_part = np.random.choice(available_tasks, size=n_random, replace=False)
            combinations[i] = np.sort(np.concatenate([forced_indices, random_part]))
        
        # Compute NITs
        batch_start_time = time()
        nits = compute_nits_batch(G, combinations)
        batch_time = time() - batch_start_time
        
        # Keep top n from batch
        keep_n = min(n_top_batteries, batch_n)
        top_indices = np.argpartition(nits, -keep_n)[-keep_n:]
        
        for idx in top_indices:
            results_list.append({
                'indices': tuple(combinations[idx]),
                'names': [condition_labels[i] for i in combinations[idx]],
                'nit': nits[idx]
            })
        
        # Progress
        elapsed = time() - start_time
        rate = batch_end / elapsed
        remaining = (n_samples - batch_end) / rate if rate > 0 else 0
        
        print(f"Processed {batch_end:,} / {n_samples:,} ({100*batch_end/n_samples:.1f}%) | "
              f"Batch: {batch_time:.2f}s | Rate: {rate:.0f}/s | ETA: {remaining:.0f}s")
    
    total_time = time() - start_time
    print(f"\nTotal time: {total_time:.1f}s ({n_samples/total_time:.0f} samples/s)")
    
    # Sort and return top results as DataFrame
    results_list.sort(key=lambda x: x['nit'], reverse=True)
    top_results = results_list[:n_top_batteries]

    df = pd.DataFrame({
        'rank': range(1, len(top_results) + 1),
        'nit': [r['nit'] for r in top_results],
        'tasks': [r['names'] for r in top_results]
    })
    return df


if __name__ == "__main__":
    atlas = 'fs32k'
    data,info = fetch_task_library(atlas= atlas,version='V1')
    full_code = info['task_code'] + '_' + info['cond_code']
    batteries = get_top_batteries(data,full_code,n_samples=1000000, battery_size=8, n_top_batteries=10,forced_tasks=None)
    print(batteries)
pass