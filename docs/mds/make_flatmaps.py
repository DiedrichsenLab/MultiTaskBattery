"""
Render a cortical (fs32k) and a cerebellar (SUIT) map PNG for every task
condition in the library, so the MDS viewer can show them on dot-click.

Each map shows the task RELATIVE TO THE MEAN ACROSS ALL TASKS (per-grayordinate
demeaning) — i.e. what is distinctive about the task, not its activation vs. rest.

  cortex      : 4-view inflated surface (L/R x lateral/medial), nilearn
  cerebellum  : SUIT flatmap, SUITPy

Offline generation step (like make_mds_data.py) — NOT needed at docs-build or
runtime; only the resulting PNGs in flatmaps/ are served.

Requires the Diedrichsen-lab toolboxes on the path (edit REPO_PATHS below):
SUITPy, surfAnalysisPy, Functional_Fusion, nitools — plus `nilearn`.
`ants` is stubbed because SUITPy imports it but the map path never uses it.

Output:  flatmaps/cortex/<full_code>.png , flatmaps/cerebellum/<full_code>.png
"""
import os, sys, json
from unittest.mock import MagicMock

# --- toolbox paths (adjust to your machine) -------------------------------
REPO_PATHS = [r'E:\nitools', r'E:\SUITPy', r'E:\surfAnalysisPy', r'E:\Functional_Fusion']
sys.modules['ants'] = MagicMock()          # SUITPy imports ants (isolation only); maps don't need it
sys.path[:0] = REPO_PATHS

import numpy as np
import pandas as pd
import nibabel as nb
import pooch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nitools as nt
import SUITPy.flatmap as fm
import surfAnalysisPy as surfpkg
from nilearn import plotting
from PIL import Image

CMAP = 'RdBu_r'
DPI = 100
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'flatmaps')


def savepng(fig, path):
    """Save with a transparent background (so the maps sit directly on the dark
    viewer, no white box), then palette-quantize keeping alpha — ~40 KB each,
    no visible loss, keeping the served flatmaps/ folder light."""
    fig.savefig(path, dpi=DPI, bbox_inches='tight', transparent=True)
    Image.open(path).convert('RGBA').quantize(
        colors=256, method=Image.FASTOCTREE, dither=Image.NONE).save(path, optimize=True)

# fs32k inflated surfaces (bundled Functional_Fusion atlas) + sulc background
FS32K_DIR = os.path.join([p for p in REPO_PATHS if p.endswith('Functional_Fusion')][0],
                         'Functional_Fusion', 'Atlases', 'tpl-fs32k')
SULC_DIR = os.path.join(os.path.dirname(surfpkg.__file__), 'standard_mesh')

# same Zenodo source / cache as make_mds_data.py
ZENODO_RECORD = "18793343"; VERSION = "V1"; ATLAS = "multiatlasHCP"
CACHE = pooch.os_cache("MTB_task_library")


def fetch():
    data = pooch.retrieve(url=f"https://zenodo.org/records/{ZENODO_RECORD}/files/desc-tasklibrary_space-{ATLAS}_{VERSION}.dscalar.nii",
                          known_hash=None, path=CACHE, fname=f"desc-tasklibrary_space-{ATLAS}_{VERSION}.dscalar.nii")
    info = pooch.retrieve(url=f"https://zenodo.org/records/{ZENODO_RECORD}/files/desc-tasklibrary_{VERSION}_info.tsv",
                          known_hash=None, path=CACHE, fname=f"desc-tasklibrary_{VERSION}_info.tsv")
    return data, info


def main():
    os.makedirs(os.path.join(OUT, 'cortex'), exist_ok=True)
    os.makedirs(os.path.join(OUT, 'cerebellum'), exist_ok=True)
    data_path, info_path = fetch()
    img = nb.load(data_path)
    codes = pd.read_csv(info_path, sep='\t')['full_code'].tolist()

    # pre-extract both structures for all conditions
    surfL, surfR = [np.asarray(s) for s in nt.surf_from_cifti(img)]          # (n, 32492) each
    cereb = nt.volume_from_cifti(img, ['cerebellum_left', 'cerebellum_right'])
    cvol = cereb.get_fdata()                                                 # (91,109,91,n)
    caffine = cereb.affine

    # Express each task RELATIVE TO THE MEAN ACROSS ALL TASKS: subtract, per
    # grayordinate, the mean over conditions. Maps then show what is distinctive
    # about a task rather than the common task-vs-rest activation shared by all.
    surfL = surfL - np.nanmean(surfL, axis=0, keepdims=True)
    surfR = surfR - np.nanmean(surfR, axis=0, keepdims=True)
    cmask = (cvol != 0).any(axis=-1)                        # voxels that belong to the cerebellum
    cvol = cvol - cvol.mean(axis=-1, keepdims=True)
    cvol[~cmask] = 0.0                                      # keep background exactly zero for ignore_zeros

    # inflated surfaces + sulc background for the cortical 4-view
    infl_L = os.path.join(FS32K_DIR, 'tpl-fs32k_hemi-L_veryinflated.surf.gii')
    infl_R = os.path.join(FS32K_DIR, 'tpl-fs32k_hemi-R_veryinflated.surf.gii')
    sulc_L = np.squeeze(np.asarray(nt.surf_from_cifti(
        nb.load(os.path.join(SULC_DIR, 'fs_L', 'fs_LR.32k.LR.sulc.dscalar.nii')), struct_names=['cortex_left'])))
    sulc_R = np.squeeze(np.asarray(nt.surf_from_cifti(
        nb.load(os.path.join(SULC_DIR, 'fs_R', 'fs_LR.32k.LR.sulc.dscalar.nii')), struct_names=['cortex_right'])))

    # Symmetric colour scales shared across ALL tasks (so maps are comparable),
    # but computed PER STRUCTURE — cerebellar signal is weaker than cortical, so
    # one global scale would leave the cerebellum washed out.
    ctx_vmax = float(np.nanpercentile(np.abs(np.concatenate([surfL, surfR], axis=1)), 98))
    cb_vmax = float(np.nanpercentile(np.abs(cvol[cmask]), 98))
    ctx_cscale = [-ctx_vmax, ctx_vmax]
    cb_cscale = [-cb_vmax, cb_vmax]
    print(f"{len(codes)} conditions | cortex cscale ±{ctx_vmax:.3f} | cerebellum cscale ±{cb_vmax:.3f}")

    # 2x2 panel layout: (left, bottom) for L-lat, R-lat, L-med, R-med
    cw, ch = 0.46, 0.50
    layout = [(0.02, 0.46, 'left', 'lateral'), (0.50, 0.46, 'right', 'lateral'),
              (0.02, -0.02, 'left', 'medial'), (0.50, -0.02, 'right', 'medial')]

    for i, code in enumerate(codes):
        # cortex — 4-view inflated surface
        fig = plt.figure(figsize=(11, 6.5))
        for left, bottom, hemi, view in layout:
            mesh = infl_L if hemi == 'left' else infl_R
            bg = sulc_L if hemi == 'left' else sulc_R
            stat = surfL[i] if hemi == 'left' else surfR[i]
            ax = fig.add_axes([left, bottom, cw, ch], projection='3d')
            plotting.plot_surf_stat_map(mesh, stat_map=stat, bg_map=bg, hemi=hemi, view=view,
                                        cmap=CMAP, vmax=ctx_vmax, colorbar=False,
                                        bg_on_data=True, alpha=1, axes=ax, figure=fig)
        savepng(fig, os.path.join(OUT, 'cortex', code + '.png'))
        plt.close(fig)

        # cerebellum — single SUIT flatmap (MNI voxels -> flatmap)
        vol = nb.Nifti1Image(cvol[..., i], caffine)
        csurf = fm.vol_to_surf(vol, space='FSL', ignore_zeros=True)
        fm.plot(csurf, cmap=CMAP, cscale=cb_cscale, new_figure=True, colorbar=False)
        savepng(plt.gcf(), os.path.join(OUT, 'cerebellum', code + '.png'))
        plt.close('all')
        print(f"  [{i+1:>2}/{len(codes)}] {code}")

    # small manifest the viewer can use to know which codes have flatmaps
    with open(os.path.join(OUT, 'index.json'), 'w') as f:
        json.dump({"codes": codes, "cmap": CMAP,
                   "cscale": {"cortex": ctx_cscale, "cerebellum": cb_cscale}}, f)
    print(f"Done -> {OUT}")


if __name__ == "__main__":
    main()
