import os
import json
import numpy as np
import pandas as pd
import nibabel as nb
import pooch

# Same Zenodo source as MultiTaskBattery/battery.py
ZENODO_RECORD = "18793343"
CACHE_DIR = pooch.os_cache("MTB_task_library")
VERSION = "V1"
ATLAS = "multiatlasHCP"

N_DIMS = 8  # MDS dimensions to keep (viewer lets you pick which 2 or 3 to plot)


def fetch_library():
    data_fname = f"desc-tasklibrary_space-{ATLAS}_{VERSION}.dscalar.nii"
    info_fname = f"desc-tasklibrary_{VERSION}_info.tsv"
    data_path = pooch.retrieve(
        url=f"https://zenodo.org/records/{ZENODO_RECORD}/files/{data_fname}",
        known_hash=None, path=CACHE_DIR, fname=data_fname)
    info_path = pooch.retrieve(
        url=f"https://zenodo.org/records/{ZENODO_RECORD}/files/{info_fname}",
        known_hash=None, path=CACHE_DIR, fname=info_fname)
    return data_path, info_path


def mds_from_centered_gram(sub, n_dims=N_DIMS):
    """Classical MDS coordinates from rest-referenced maps `sub` (n_cond x n_chan)."""
    n = sub.shape[0]
    G = sub @ sub.T                          # second-moment / Gram (NIT's matrix)
    H = np.eye(n) - np.ones((n, n)) / n
    B = H @ G @ H                            # double-center across conditions
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    coords = V[:, :n_dims] * np.sqrt(np.clip(w[:n_dims], 0, None))
    pos = w[w > 1e-9]
    var_explained = (w[:n_dims] / pos.sum()) if pos.size else np.zeros(n_dims)
    return coords, var_explained


def main():
    data_path, info_path = fetch_library()
    img = nb.load(data_path)
    data = img.get_fdata()
    info = pd.read_csv(info_path, sep="\t")
    labels = info["full_code"].tolist()
    task_code = info["task_code"].tolist()  # used for color grouping in the viewer
    cond_code = info["cond_code"].tolist()  # condition; "task" means no real condition
    n_chan = data.shape[1]

    out = {
        "metric": "centered_gram_euclidean",
        "version": VERSION,
        "atlas": ATLAS,
        "labels": labels,
        "task_code": task_code,
        "cond_code": cond_code,
        "structures": {},
    }

    # Collect the column range of every base CIFTI structure
    brain_models = img.header.get_axis(1)
    base_ranges = {}  # short name -> (start, end)
    for name, idx, _ in brain_models.iter_structures():
        s = idx.start
        e = idx.stop if idx.stop is not None else n_chan
        base_ranges[name.replace("CIFTI_STRUCTURE_", "")] = (s, e)

    def add(key, ranges):
        """Compute MDS over the greyordinates spanned by `ranges` (list of (s,e))."""
        sub = np.concatenate([data[:, s:e] for s, e in ranges], axis=1)
        coords, var = mds_from_centered_gram(sub)
        out["structures"][key] = {
            "coords": np.round(coords, 4).tolist(),
            "var_explained": np.round(var, 4).tolist(),
            "n_channels": int(sub.shape[1]),
        }

    # 1) Each hemisphere/structure separately (e.g. CORTEX_LEFT)
    for key, rng in base_ranges.items():
        add(key, [rng])

    # 2) Bilateral structures combined (e.g. CORTEX = LEFT + RIGHT)
    combined = {}
    for key, rng in base_ranges.items():
        base = key.replace("_LEFT", "").replace("_RIGHT", "")
        combined.setdefault(base, []).append(rng)
    for base, ranges in combined.items():
        if len(ranges) > 1:                      # only emit if there were L/R parts to merge
            add(base, ranges)

    # 3) Whole brain (all greyordinates)
    add("WHOLE_BRAIN", list(base_ranges.values()))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mds_data.json")
    with open(out_path, "w") as f:
        json.dump(out, f)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"Wrote {out_path} ({size_kb:.1f} KB): "
          f"{len(out['structures'])} structures x {len(labels)} conditions")


if __name__ == "__main__":
    main()
