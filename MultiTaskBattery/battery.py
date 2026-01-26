import os
import numpy as np
import pandas as pd
import nibabel as nb
import pooch
from numba import njit
from time import time


GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
REPO_URL = "https://api.github.com/repos/Barafat2/MTB_task_library/contents/"
CACHE_DIR = pooch.os_cache("MTB_task_library")


def fetch_task_library(version='V1',atlas = 'multiatlasHCP', structures=None):
    """
    Fetch task activation library from zenodo

    Args:
        version: Library version (e.g., 'V1')
        structures: List of CIFTI structure names to include. If None, loads all.

    Returns:
        library_data: ndarray (n_conditions, n_measurement_channels)
        library_info: DataFrame with condition info
    """
    if GITHUB_TOKEN is None:
        raise ValueError("GITHUB_TOKEN environment variable not set")

    data_file = f"{version}/desc-tasklibrary_space-{atlas}_{version}.dscalar.nii"
    info_file = f"{version}/desc-tasklibrary_{version}_info.tsv"

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

    return load_library(data_path, info_path, structures=structures)


def load_library(data_path, info_path, structures=None):
    """
    Load task library_data from local files.

    Args:
        data_path: Path to library data file (.nii, .nii.gz, .dscalar.nii), must be shape (n_conditions, n_measurement_channels)
        info_path: Path to library info file (.tsv) with 'full_code' column
        structures: List of CIFTI structure names to include (e.g., ['CIFTI_STRUCTURE_CORTEX_LEFT',
                    'CIFTI_STRUCTURE_CORTEX_RIGHT']). If None, loads all data.

    Returns:
        library_data: ndarray (n_conditions, n_measurement_channels)
        library_info: DataFrame with condition info
    """
    img = nb.load(data_path)
    library_data = img.get_fdata()
    library_info = pd.read_csv(info_path, sep='\t')

    if 'full_code' not in library_info.columns:
        raise ValueError("info file must have 'full_code' column")

    # Filter by structures if specified (CIFTI files only)
    if structures is not None:
        brain_models = img.header.get_axis(1)
        
        # Convert simple names to CIFTI structure names
        cifti_structures = []
        for s in structures:
            cifti_name = nb.cifti2.CIFTI_BRAIN_STRUCTURES.get(s.upper(), s)
            cifti_structures.append(cifti_name)
        
        indices = []
        for bm in brain_models.iter_structures():
            struct_name, _, idx = bm
            if struct_name in cifti_structures:
                indices.extend(range(idx.start, idx.stop))
        if not indices:
            available = [bm[0] for bm in brain_models.iter_structures()]
            raise ValueError(f"No matching structures found. Available: {available}")
        library_data = library_data[:, indices]
    
    print(f"Loaded {library_data.shape[0]} conditions, {library_data.shape[1]} greyordinates")
    return library_data, library_info


@njit
def _compute_nit(G):
    """
    Computes negative inverse trace of the battery task-by-task covariance matrix
    """
    N = G.shape[0]

    # Center the matrix
    H = np.eye(N) - np.ones((N, N)) / N
    G_mc = H @ G @ H

    # Eigenvalues
    l_mc = np.linalg.eigvalsh(G_mc)

    # Negative inverse trace
    l_mc = l_mc[::-1]
    l_mc[l_mc < 1e-12] = 1e-12
    nit = -np.sum(1 / l_mc)
    return nit


def evaluate_battery(library_data, library_info, battery_full_codes):
    """
    Evaluate a task battery by computing the negative inverse trace (NIT)
    of the task-by-task covariance matrix.

    Args:
        library_data: Task activation data (n_conditions, n_measurement_channels)
        library_info: DataFrame with condition info (must have 'full_code' column)
        battery_full_codes: List of task condition codes from the 'full_code' column in the info file (e.g., ['task1_cond1', 'task2_cond2'])

    Returns: NIT score (float) - higher values indicate more separable task patterns and more optimal battery
    """
    full_codes = library_info['full_code'].str.lower().tolist()
    comb_names_lower = [name.lower() for name in battery_full_codes]

    # Check for missing tasks
    missing = [name for name in comb_names_lower if name not in full_codes]
    if missing:
        raise ValueError(f"Tasks not found in info: {missing}")

    idx = np.array([full_codes.index(name) for name in comb_names_lower])
    sub_data = library_data[idx]
    G = sub_data @ sub_data.T
    return _compute_nit(G)


def get_top_batteries(library_data, library_info, n_samples,
                    battery_size=8, n_top_batteries=10, forced_tasks=None,
                    verbose=True):
    """
    Random search over task combinations to find highest NIT.

    Args:
        library_data: Task activation data (n_conditions, n_measurement_channels)
        library_info: DataFrame with condition info (must have 'full_code' column)
        n_samples: Total number of random batteries to test
        battery_size: Number of task conditions in each battery
        n_top_batteries: Number of top batteries to keep
        forced_tasks: List of task names to include in every battery

    Returns: DataFrame with columns 'rank', 'evaluation', 'battery'
    """
    condition_labels = library_info['full_code'].tolist()
    n_lib_task = library_data.shape[0]

    # Compute full G matrix once
    G_full = library_data @ library_data.T

    # Handle forced tasks - convert labels to indices
    if forced_tasks is None:
        forced_tasks = []
    label_to_idx = {label: i for i, label in enumerate(condition_labels)}
    forced_indices = np.array([label_to_idx[t] for t in forced_tasks], dtype=np.int64)
    n_forced = len(forced_indices)
    n_random = battery_size - n_forced

    # Available tasks for random selection (exclude forced)
    available_tasks = np.array([i for i in range(n_lib_task) if i not in forced_indices])

    # Track top N results
    top_results = []
    min_nit = float('-inf')

    if verbose:
        print(f"Starting computation... (forced: {n_forced}, random: {n_random})")
    start_time = time()

    for i in range(n_samples):
        # Generate random combination
        random_part = np.random.choice(available_tasks, size=n_random, replace=False)
        idx = np.sort(np.concatenate([forced_indices, random_part]))

        # Compute NIT
        G = G_full[np.ix_(idx, idx)]
        nit = _compute_nit(G)

        # Keep top N
        if len(top_results) < n_top_batteries:
            top_results.append((nit, tuple(idx)))
            if len(top_results) == n_top_batteries:
                top_results.sort()
                min_nit = top_results[0][0]
        elif nit > min_nit:
            top_results[0] = (nit, tuple(idx))
            top_results.sort()
            min_nit = top_results[0][0]

        if verbose:
            # Progress updates every 10%
            if (i + 1) % (n_samples // 10) == 0:
                elapsed = time() - start_time
                print(f"Processed {i + 1:,} / {n_samples:,} ({100 * (i + 1) / n_samples:.0f}%) | "
                    f"Rate: {(i + 1) / elapsed:.0f}/s")

    # Sort results by NIT descending
    top_results.sort(reverse=True)

    df = pd.DataFrame({
        'rank': range(1, len(top_results) + 1),
        'evaluation': [r[0] for r in top_results],
        'battery': [[condition_labels[i] for i in r[1]] for r in top_results]
    })
    return df


if __name__ == "__main__":
    structures = None
    data, info = fetch_task_library(version='V1',structures=structures)
    batteries = get_top_batteries(data, info, n_samples=1000000, battery_size=8, n_top_batteries=10, forced_tasks=None)
    print(batteries)